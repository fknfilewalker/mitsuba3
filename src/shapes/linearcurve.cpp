#include <mitsuba/core/fwd.h>
#include <mitsuba/core/math.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/mmap.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/interaction.h>
#include <mitsuba/render/shape.h>
#include <mitsuba/render/scene_ir.h>

#include <drjit/texture.h>

#if defined(MI_ENABLE_EMBREE)
#include <embree3/rtcore.h>
#endif

#if defined(MI_ENABLE_METAL)
#include "../render/metal/shapes.h"
#endif

#include "../render/bbox_reduce.h"

NAMESPACE_BEGIN(mitsuba)

/**!

.. _shape-linearcurve:

Linear curve (:monosp:`linearcurve`)
-------------------------------------------------

.. pluginparameters::
 :extra-rows: 2

 * - filename
   - |string|
   - Filename of the curves to be loaded

 * - to_world
   - |transform|
   - Specifies a linear object-to-world transformation. Note that the control
     points' raddii are invariant to this transformation!

 * - control_point_count
   - |int|
   - Total number of control points
   - |exposed|

 * - segment_indices
   - :paramtype:`uint32[]`
   - Starting indices of a linear segment
   - |exposed|

 * - control_points
   - :paramtype:`float[]`
   - Flattened control points buffer pre-multiplied by the object-to-world transformation.
     Each control point in the buffer is structured as follows: position_x, position_y, position_z, radius
   - |exposed|

.. subfigstart::
.. subfigure:: ../../resources/data/docs/images/render/shape_linearcurve_basic.jpg
   :caption: Basic example
.. subfigure:: ../../resources/data/docs/images/render/shape_linearcurve_parameterization.jpg
   :caption: A textured linear curve with the default parameterization
.. subfigend::
   :label: fig-linearcurve

This shape plugin describes multiple linear curves. They are hollow
cylindrical tubes which can have varying radii along their length. The linear
segments are connected by a smooth spherical joint, and they are also
terminated by a spherical endcap. This shape should always be preferred over
curve approximations modeled using triangles.

Although it is possible to define multiple curves as multiple separate objects,
this plugin was intended to be used as an aggregate of curves. Of course,
if the individual curves need different materials or other individual
characteristics they need to be defined in separate objects.

The file from which curves are loaded defines a single control point per line
using four real numbers. The first three encode the position and the last one is
the radius of the control point. At least two control points need to be
specified for a single curve. Empty lines between control points are used to
indicate the beginning of a new curve. Here is an example of two curves, the
first with 2 control points and static radii and the second with 4 control
points and increasing radii::

    -1.0 0.1 0.1 0.5
     1.0 1.4 1.2 0.5

    -1.0 5.0 2.2 1
    -2.3 4.0 2.3 2
     4.0 1.0 2.2 5
     4.0 0.0 2.3 6

.. tabs::
    .. code-tab:: xml
        :name: linearcurve

        <shape type="linearcurve">
            <transform name="to_world">
                <translate x="1" y="0" z="0"/>
                <scale value="2"/>
            </transform>
            <string name="filename" type="curves.txt"/>
        </shape>

    .. code-tab:: python

        'curves': {
            'type': 'linearcurve',
            'to_world': mi.ScalarAffineTransform4f().scale([2, 2, 2]).translate([1, 0, 0]),
            'filename': 'curves.txt'
        },

.. note:: The backfaces of curves are always culled. It is therefore impossible
          to intersect the curve with a ray that's origin is inside of the curve.

.. note:: This shape supports differentiable rendering: the ``control_points``
          are differentiable, and discontinuous (visibility) derivatives are
          handled for the smooth silhouette of the conical segment bodies (and
          their spherical joints). The silhouette contribution of the *spherical
          end caps* is not yet modeled, so primarily-visible boundary gradients
          may be slightly underestimated near the rounded extremities of a curve.
*/

template <typename Float, typename Spectrum>
class LinearCurve final : public Shape<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Shape, m_to_world, m_is_instance, m_discontinuity_types,
                   m_shape_type, initialize, mark_dirty, get_children_string,
                   parameters_grad_enabled)
    MI_IMPORT_TYPES()

    using typename Base::ScalarIndex;
    using typename Base::ScalarSize;

    using InputFloat = float;
    using InputPoint3f  = Point<InputFloat, 3>;
    using InputVector3f = Vector<InputFloat, 3>;
    using FloatStorage = DynamicBuffer<dr::replace_scalar_t<Float, InputFloat>>;
    using UInt32Storage = DynamicBuffer<UInt32>;
    using Index = typename CoreAliases::UInt32;

    LinearCurve(const Properties &props) : Base(props) {
#if !defined(MI_ENABLE_EMBREE)
        if constexpr (!dr::is_jit_v<Float>)
            Throw("The linear curve is only available with Embree in scalar "
                  "variants!");
#endif

        auto fs = file_resolver();
        fs::path file_path = fs->resolve(props.get<std::string_view>("filename"));
        std::string m_name = file_path.filename().string();

        // used for throwing an error later
        auto fail = [&](const char *descr, auto... args) {
            Throw(("Error while loading linear curve(s) from \"%s\": " + std::string(descr))
                      .c_str(), m_name, args...);
        };

        Log(Debug, "Loading linear curve(s) from \"%s\" ..", m_name);
        if (!fs::exists(file_path))
            fail("file not found!");

        ref<MemoryMappedFile> mmap = new MemoryMappedFile(file_path);
        ScopedPhase phase(ProfilerPhase::LoadGeometry);

        // Temporary buffers for vertices and radius
        std::vector<InputPoint3f> vertices;
        std::vector<InputFloat> radius;
        ScalarSize vertex_guess = (ScalarSize) mmap->size() / 100;
        vertices.reserve(vertex_guess);
        radius.reserve(vertex_guess);

        // Load data from the given file
        const char *ptr = (const char *) mmap->data();
        const char *eof = ptr + mmap->size();
        char buf[1025];
        Timer timer;

        size_t segment_count = 0;
        std::vector<size_t> curve_1st_idx;
        curve_1st_idx.reserve(vertex_guess / 4);
        bool new_curve = true;

        auto finish_curve = [&]() {
            if (!new_curve) {
                size_t num_control_points = vertices.size() - curve_1st_idx[curve_1st_idx.size() - 1];
                if (unlikely((num_control_points < 2) && (num_control_points > 0)))
                    fail("Linear curves must have at least two control points!");
                if (likely(num_control_points > 0))
                    segment_count += (num_control_points - 1);
            }
        };

        while (ptr < eof) {
            // Determine the offset of the next newline
            const char *next = ptr;
            advance<false>(&next, eof, "\n");

            // Copy buf into a 0-terminated buffer
            ScalarSize size = (ScalarSize) (next - ptr);
            if (size >= sizeof(buf) - 1)
                fail("file contains an excessively long line! (%i characters)!", size);
            memcpy(buf, ptr, size);
            buf[size] = '\0';

            // Skip whitespace(s)
            const char *cur = buf, *eol = buf + size;
            advance<true>(&cur, eol, " \t\r");
            bool parse_error = false;

            // Empty line
            if (*cur == '\0') {
                finish_curve();
                new_curve = true;
                ptr = next + 1;
                continue;
            }

            // Handle current line: v.x v.y v.z radius
            if (new_curve) {
                curve_1st_idx.push_back(vertices.size());
                new_curve = false;
            }

            // Vertex position
            InputPoint3f p;
            for (ScalarSize i = 0; i < 3; ++i) {
                const char *orig = cur;
                p[i] = string::strtof<InputFloat>(cur, (char **) &cur);
                parse_error |= cur == orig;
            }
            p = m_to_world.scalar() * p;

            // Vertex radius
            InputFloat r;
            const char *orig = cur;
            r = string::strtof<InputFloat>(cur, (char **) &cur);
            parse_error |= cur == orig;

            if (unlikely(!all(dr::isfinite(p))))
                fail("Control point contains invalid position data (line: \"%s\")!", buf);
            if (unlikely(!dr::isfinite(r)))
                fail("Control point contains invalid radius data (line: \"%s\")!", buf);

            vertices.push_back(p);
            radius.push_back(r);

            if (unlikely(parse_error))
                fail("Could not parse line \"%s\"!", buf);
            ptr = next + 1;
        }
        if (curve_1st_idx.size() == 0)
            fail("Empty curve file: no control points were read!");
        finish_curve();

        m_control_point_count = (ScalarSize) vertices.size();

        std::unique_ptr<ScalarIndex[]> indices = std::make_unique<ScalarIndex[]>(segment_count);
        std::unique_ptr<ScalarIndex[]> curves_1st_prim_idx =
            std::make_unique<ScalarIndex[]>(curve_1st_idx.size() + 1);
        size_t segment_index = 0;
        for (size_t i = 0; i < curve_1st_idx.size(); ++i) {
            size_t next_curve_idx = i + 1 < curve_1st_idx.size() ? curve_1st_idx[i + 1] : vertices.size();
            size_t curve_segment_count = next_curve_idx - curve_1st_idx[i] - 1;
            curves_1st_prim_idx[i] = (ScalarIndex) segment_index;
            for (size_t j = 0; j < curve_segment_count; ++j)
                indices[segment_index++] = (ScalarIndex) (curve_1st_idx[i] + j);
        }
        curves_1st_prim_idx[curve_1st_idx.size()] = (ScalarIndex) segment_index;
        m_indices = dr::load<UInt32Storage>(indices.get(), segment_count);
        m_curves_prim_idx = dr::load<UInt32Storage>(curves_1st_prim_idx.get(),
                                                    curve_1st_idx.size() + 1);

        std::unique_ptr<InputFloat[]> positions =
            std::make_unique<InputFloat[]>(m_control_point_count * 3);
        for (ScalarIndex i = 0; i < vertices.size(); i++) {
            InputFloat *vertex_ptr = positions.get() + i * 3;
            dr::store(vertex_ptr, vertices[i]);
        }

        // Merge buffers into m_control_points
        m_control_points = dr::empty<FloatStorage>(m_control_point_count * 4);
        FloatStorage vertex_buffer = dr::load<FloatStorage>(positions.get(), m_control_point_count * 3);
        FloatStorage radius_buffer = dr::load<FloatStorage>(radius.data(), m_control_point_count * 1);

        if constexpr (dr::is_jit_v<Float>) {
            DynamicBuffer<UInt32> idx = dr::arange<DynamicBuffer<UInt32>>(m_control_point_count);
            dr::scatter(m_control_points, dr::gather<FloatStorage>(vertex_buffer, idx * 3u + 0u), idx * 4u + 0u);
            dr::scatter(m_control_points, dr::gather<FloatStorage>(vertex_buffer, idx * 3u + 1u), idx * 4u + 1u);
            dr::scatter(m_control_points, dr::gather<FloatStorage>(vertex_buffer, idx * 3u + 2u), idx * 4u + 2u);
            dr::scatter(m_control_points, dr::gather<FloatStorage>(radius_buffer, idx * 1u + 0u), idx * 4u + 3u);
        } else {
            for (size_t i = 0; i < m_control_point_count; ++i) {
                m_control_points[i * 4 + 0] = vertex_buffer[i * 3 + 0];
                m_control_points[i * 4 + 1] = vertex_buffer[i * 3 + 1];
                m_control_points[i * 4 + 2] = vertex_buffer[i * 3 + 2];
                m_control_points[i * 4 + 3] = radius_buffer[i * 1 + 0];
            }
        }

        // Compute bounding box
        m_bbox.reset();
        for (ScalarSize i = 0; i < m_control_point_count; ++i) {
            ScalarPoint3f p(positions[3 * i + 0], positions[3 * i + 1],
                            positions[3 * i + 2]);
            ScalarFloat r(radius[i]);
            m_bbox.expand(p + r * ScalarVector3f(-1, 0, 0));
            m_bbox.expand(p + r * ScalarVector3f(1, 0, 0));
            m_bbox.expand(p + r * ScalarVector3f(0, -1, 0));
            m_bbox.expand(p + r * ScalarVector3f(0, 1, 0));
            m_bbox.expand(p + r * ScalarVector3f(0, 0, -1));
            m_bbox.expand(p + r * ScalarVector3f(0, 0, 1));
        }

        ScalarSize control_point_bytes = 4 * sizeof(InputFloat);
        Log(Debug, "\"%s\": read %i control points (%s in %s)",
            m_name, m_control_point_count,
            util::mem_string(m_control_point_count * control_point_bytes),
            util::time_string((float) timer.value())
        );

        m_shape_type = ShapeType::LinearCurve;

        // The linear curve is watertight (spherical joints and endcaps), so it
        // has no open boundary: only smooth (interior) silhouettes occur.
        m_discontinuity_types = (uint32_t) DiscontinuityFlags::InteriorType;

        initialize();
    }

    ScalarSize primitive_count() const override { return (ScalarSize) dr::width(m_indices); }

    SurfaceInteraction3f compute_surface_interaction(const Ray3f &ray,
                                                     const PreliminaryIntersection3f &pi,
                                                     uint32_t ray_flags,
                                                     uint32_t recursion_depth,
                                                     Mask active) const override {
        MI_MASK_ARGUMENT(active);
        constexpr bool IsDiff = dr::is_diff_v<Float>;

        // Early exit when tracing isn't necessary
        if (!m_is_instance && recursion_depth > 0)
            return dr::zeros<SurfaceInteraction3f>();

        bool need_dn_duv = has_flag(ray_flags, RayFlags::dNSdUV) ||
                           has_flag(ray_flags, RayFlags::dNGdUV);
        bool need_dp_duv = has_flag(ray_flags, RayFlags::dPdUV) || need_dn_duv;
        bool need_uv     = has_flag(ray_flags, RayFlags::UV) || need_dp_duv;
        bool detach_shape = has_flag(ray_flags, RayFlags::DetachShape);
        bool follow_shape = has_flag(ray_flags, RayFlags::FollowShape);

        dr::suspend_grad<Float> scope(detach_shape, m_control_points);

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();

        Float v_local = pi.prim_uv.x();
        UInt32 prim_idx = pi.prim_index;

        // The `v_local` reported by the ray tracer is shifted along the segment
        // such that the surface normal is `normalize(p - c)` with
        // `c = lerp(cp0, cp1, v_local)` (this absorbs the cone tilt induced by
        // the changing radius).
        Point3f c;
        Vector3f axis, u_rot, u_rad;
        Float radius, dr_dv, sin_a, cos_a;
        std::tie(c, axis, std::ignore, radius, dr_dv, u_rot, u_rad, sin_a, cos_a) =
            cone_geometry(v_local, prim_idx, active);

        if constexpr (IsDiff) {
            // Attached interaction point (w.r.t. the control points)
            Point3f p = ray(pi.t);
            Vector3f rad_vec = dr::normalize(p - c);
            Float u = dr::atan2(dr::dot(u_rot, rad_vec),
                                dr::dot(u_rad, rad_vec));
            u += dr::select(u < 0.f, dr::TwoPi<Float>, 0.f);
            u *= dr::InvTwoPi<Float>;
            u = dr::detach(u); // u has no motion

            auto [sin_u, cos_u] = dr::sincos(u * dr::TwoPi<Float>);
            Vector3f rad = cos_u * u_rad + sin_u * u_rot;
            Normal3f n = cos_a * rad + sin_a * axis;
            Point3f p_diff = c + radius * n;
            p = dr::replace_grad(p, p_diff);

            if (follow_shape) {
                si.p = p;
                Float t_diff = dr::sqrt(dr::squared_norm(si.p - ray.o) /
                                        dr::squared_norm(ray.d));
                si.t = dr::replace_grad(pi.t, t_diff);
            } else {
                // Keep the point on the ray by intersecting the (differentiable)
                // tangent plane through `p` with normal `n`.
                Float t_diff = dr::dot(p - ray.o, n) / dr::dot(n, ray.d);
                si.t = dr::replace_grad(pi.t, t_diff);
                si.p = ray(si.t);

                // Recover `v_local`'s motion along the segment
                Float v_global = (v_local + prim_idx) / dr::width(m_indices);
                Vector3f dp_dv;
                std::tie(std::ignore, dp_dv, std::ignore, std::ignore,
                         std::ignore, std::ignore, std::ignore) =
                    partials(Point2f(u, v_global), active);
                dp_dv = dr::detach(dp_dv);
                Float v_diff = dr::dot(si.p - p_diff, dp_dv) /
                               dr::squared_norm(dp_dv);
                v_global = dr::replace_grad(v_global, v_diff);
                Float v_local_diff = v_global * dr::width(m_indices) - prim_idx;
                v_local = dr::replace_grad(v_local, v_local_diff);

                std::tie(c, axis, std::ignore, radius, dr_dv, u_rot, u_rad, sin_a, cos_a) =
                    cone_geometry(v_local, prim_idx, active);
            }
        } else {
            si.t = pi.t;
            si.p = ray(si.t);
        }

        si.t = dr::select(active, si.t, dr::Infinity<Float>);

        // Normal
        Normal3f n = dr::normalize(si.p - c);
        si.n = si.sh_frame.n = n;

        // Embree and OptiX cull linear-curve backfaces at trace time; Metal's
        // HW intersector reports both sides. Drop inside hits to match (a no-op
        // on backends that already cull).
        this->cull_backface(si, ray, active);

        if (need_uv) {
            Vector3f rad_vec = dr::normalize(si.p - c);
            Float u = dr::atan2(dr::dot(u_rot, rad_vec),
                                dr::dot(u_rad, rad_vec));
            u += dr::select(u < 0.f, dr::TwoPi<Float>, 0.f);
            u *= dr::InvTwoPi<Float>;
            Float v = (v_local + prim_idx) / dr::width(m_indices);

            si.uv = Point2f(u, v);
        }

        if (need_dp_duv) {
            Vector3f dp_du, dp_dv, dn_du, dn_dv;
            std::tie(dp_du, dp_dv, dn_du, dn_dv, std::ignore, std::ignore,
                     std::ignore) = partials(si.uv, active);
            si.dp_du = dp_du;
            si.dp_dv = dp_dv;
            if (need_dn_duv) {
                si.dn_du = dn_du;
                si.dn_dv = dn_dv;
            }
        }

        si.prim_index = pi.prim_index;
        si.shape    = this;
        si.instance = nullptr;

        return si;
    }

    SurfaceInteraction3f eval_parameterization(const Point2f &uv,
                                               uint32_t ray_flags,
                                               Mask active) const override {
        PreliminaryIntersection3f pi = dr::zeros<PreliminaryIntersection3f>();
        Float eps = dr::Epsilon<Float>;

        Float v_global = uv.y();
        size_t segment_count = dr::width(m_indices);
        UInt32 segment_idx = dr::floor2int<UInt32>(v_global * segment_count);
        segment_idx = dr::clip(segment_idx, 0, (uint32_t) segment_count - 1);
        Float v_local = v_global * segment_count - segment_idx;

        pi.prim_uv.x() = v_local;
        pi.prim_uv.y() = 0;
        pi.prim_index = segment_idx;
        pi.shape = this;
        pi.valid = active;
        dr::masked(pi.t, active) = eps * 10;

        auto [c, axis, L, radius, dr_dv, u_rot, u_rad, sin_a, cos_a] =
            cone_geometry(v_local, segment_idx, active);
        auto [sin_u, cos_u] = dr::sincos(uv.x() * dr::TwoPi<Float>);
        Vector3f rad = cos_u * u_rad + sin_u * u_rot;
        Normal3f n = cos_a * rad + sin_a * axis;
        Point3f p = c + radius * n;

        // Offset the point slightly outward and shoot a ray back at it so that
        // `compute_surface_interaction` recovers the same point/normal.
        Point3f o = p + n * pi.t;
        Ray3f ray(o, -n, 0, Wavelength(0));

        SurfaceInteraction3f si =
            compute_surface_interaction(ray, pi, ray_flags, 0, active);
        si.finalize_surface_interaction(pi, ray, ray_flags, active);

        return si;
    }

    Point3f differential_motion(const SurfaceInteraction3f &si,
                                Mask active) const override {
        MI_MASK_ARGUMENT(active);

        if constexpr (!dr::is_diff_v<Float>) {
            return si.p;
        } else {
            Point2f uv = dr::detach(si.uv);

            size_t segment_count = dr::width(m_indices);
            UInt32 seg = dr::floor2int<UInt32>(uv.y() * segment_count);
            seg = dr::clip(seg, 0, (uint32_t) segment_count - 1);
            Float v_local = uv.y() * segment_count - seg;

            auto [c, axis, L, radius, dr_dv, u_rot, u_rad, sin_a, cos_a] =
                cone_geometry(v_local, seg, active);
            auto [sin_u, cos_u] = dr::sincos(uv.x() * dr::TwoPi<Float>);
            Vector3f rad = cos_u * u_rad + sin_u * u_rot;
            Normal3f n = cos_a * rad + sin_a * axis;
            Point3f p_diff = c + radius * n;

            return dr::replace_grad(si.p, p_diff);
        }
    }

    // =============================================================
    //! @{ \name Silhouette sampling routines (smooth/interior only)
    // =============================================================

    SilhouetteSample3f sample_silhouette(const Point3f &sample,
                                         uint32_t flags,
                                         Mask active) const override {
        MI_MASK_ARGUMENT(active);
        SilhouetteSample3f ss = dr::zeros<SilhouetteSample3f>();

        if (!has_flag(flags, DiscontinuityFlags::InteriorType))
            return ss;

        // Sample a point on the surface
        ss.uv = Point2f(sample.y(), sample.x());
        auto [dp_du, dp_dv, dn_du, dn_dv, L, M, N] = partials(ss.uv, active);
        SurfaceInteraction3f si = eval_parameterization(
            ss.uv, +RayFlags::AllNonDifferentiable, active);
        ss.p = si.p;

        // Sample a tangential direction at the point
        ss.d = warp::interval_to_tangent_direction(si.n, sample.z());

        ss.discontinuity_type = (uint32_t) DiscontinuityFlags::InteriorType;
        ss.flags = flags;
        ss.n = si.n;

        Float E = dr::squared_norm(dp_du),
              F = dr::dot(dp_du, dp_dv),
              G = dr::squared_norm(dp_dv);
        Float det_I = E * G - F * F;
        ss.pdf = dr::safe_rsqrt(det_I); // area element ratio
        ss.pdf *= dr::InvTwoPi<Float>;

        Float a = dr::dot(ss.d, dp_du) / E,
              b = dr::dot(ss.d, dp_dv) / G;
        ss.silhouette_d =
            dr::normalize(dr::cross(ss.n, a * dn_du + b * dn_dv));
        ss.foreshortening =
            dr::abs((a * a * L + 2 * a * b * M + b * b * N) /
                    (a * a * E + 2 * a * b * F + b * b * G));

        ss.shape = this;
        ss.offset = silhouette_offset;
        return ss;
    }

    Point3f invert_silhouette_sample(const SilhouetteSample3f &ss,
                                     Mask active) const override {
        MI_MASK_ARGUMENT(active);
        Point3f sample = dr::zeros<Point3f>();
        sample.z() = warp::tangent_direction_to_interval(ss.n, ss.d);
        sample.y() = ss.uv.x();
        sample.x() = ss.uv.y();
        return sample;
    }

    SilhouetteSample3f primitive_silhouette_projection(const Point3f &viewpoint,
                                                       const SurfaceInteraction3f &si,
                                                       uint32_t flags,
                                                       Float /*sample*/,
                                                       Mask active) const override {
        MI_MASK_ARGUMENT(active);
        SilhouetteSample3f ss = dr::zeros<SilhouetteSample3f>();

        if (has_flag(flags, DiscontinuityFlags::InteriorType)) {
            size_t segment_count = dr::width(m_indices);
            UInt32 segment_id = dr::floor2int<UInt32>(si.uv.y() * segment_count);
            segment_id = dr::clip(segment_id, 0, (uint32_t) segment_count - 1);
            Float v_local = si.uv.y() * segment_count - segment_id;

            Point3f c;
            Vector3f axis, u_rot, u_rad;
            Float radius, sin_a, cos_a;
            std::tie(c, axis, std::ignore, radius, std::ignore, u_rot, u_rad,
                     sin_a, cos_a) = cone_geometry(v_local, segment_id, active);

            Vector3f OC = c - viewpoint;
            Float OC_norm = dr::norm(OC);
            OC /= OC_norm;

            // Find a silhouette point by fixing `si.v` (along the curve) and
            // bisecting `si.u`. The smooth silhouette condition is that the cone
            // normal `N(u)` is perpendicular to the view direction:
            //   dot(N(u), OC) + radius / |OC| = 0
            const auto normal_eq = [&](Float u) {
                auto [sin_u, cos_u] = dr::sincos(u * dr::TwoPi<Float>);
                Vector3f rad = cos_u * u_rad + sin_u * u_rot;
                Normal3f n = cos_a * rad + sin_a * axis;
                return dr::dot(n, OC) + radius / OC_norm;
            };
            Float u_lower = si.uv.x() - 0.25f + math::ShadowEpsilon<Float>,
                  u_upper = si.uv.x() + 0.25f - math::ShadowEpsilon<Float>;
            Float f_lower = normal_eq(u_lower),
                  f_upper = normal_eq(u_upper);

            Mask success = active & (f_lower * f_upper < 0.f),
                 active_loop = Mask(success);
            UInt32 cnt = 0u;

            std::tie(u_lower, u_upper, f_lower, f_upper, cnt, active_loop) =
                dr::while_loop(
                std::make_tuple(u_lower, u_upper, f_lower, f_upper, cnt, active_loop),
                [](const Float &, const Float &, const Float &, const Float &,
                   const UInt32 &, const Mask &active_loop) { return active_loop; },
                [normal_eq](Float &u_lower, Float &u_upper, Float &f_lower,
                            Float &f_upper, UInt32 &cnt, Mask &active_loop) {
                    Float u_middle = 0.5f * (u_lower + u_upper);
                    Float f_middle = normal_eq(u_middle);
                    Mask lower = f_middle * f_lower <= 0.f;
                    u_lower = dr::select(lower, u_lower, u_middle);
                    u_upper = dr::select(lower, u_middle, u_upper);
                    f_lower = dr::select(lower, f_lower, f_middle);
                    f_upper = dr::select(lower, f_middle, f_upper);
                    cnt += 1u;
                    active_loop &= cnt < 22u;
                },
                "Linear curve projection bisection");

            ss.discontinuity_type =
                dr::select(success,
                           (uint32_t) DiscontinuityFlags::InteriorType,
                           (uint32_t) DiscontinuityFlags::Empty);

            dr::masked(u_lower, u_lower < 0.f) += 1.f;
            dr::masked(u_lower, u_lower > 1.f) -= 1.f;

            ss.uv = Point2f(u_lower, si.uv.y());
            SurfaceInteraction3f si_ = eval_parameterization(
                ss.uv, +RayFlags::AllNonDifferentiable, active);
            ss.p = si_.p;
            ss.n = si_.n;
            ss.d = dr::normalize(ss.p - viewpoint);
            ss.prim_index = si_.prim_index;

            Vector3f dp_du, dp_dv, dn_du, dn_dv;
            std::tie(dp_du, dp_dv, dn_du, dn_dv, std::ignore, std::ignore,
                     std::ignore) = partials(ss.uv, active);
            Float E = dr::squared_norm(dp_du),
                  G = dr::squared_norm(dp_dv);
            Float a = dr::dot(ss.d, dp_du) / E,
                  b = dr::dot(ss.d, dp_dv) / G;
            ss.silhouette_d =
                dr::normalize(dr::cross(ss.n, a * dn_du + b * dn_dv));
        }

        ss.flags = flags;
        ss.shape = this;
        ss.offset = silhouette_offset;
        return ss;
    }

    std::tuple<DynamicBuffer<UInt32>, DynamicBuffer<Float>>
    precompute_silhouette(const ScalarPoint3f & /*viewpoint*/) const override {
        std::vector<uint32_t> type = { +DiscontinuityFlags::InteriorType };
        std::vector<ScalarFloat> weight = { 1.f };
        return { dr::load<DynamicBuffer<UInt32>>(type.data(), type.size()),
                 dr::load<DynamicBuffer<Float>>(weight.data(), weight.size()) };
    }

    SilhouetteSample3f
    sample_precomputed_silhouette(const Point3f &viewpoint,
                                  UInt32 /*sample1*/, Float sample2,
                                  Mask active) const override {
        MI_MASK_ARGUMENT(active);

        SurfaceInteraction3f si = dr::zeros<SurfaceInteraction3f>();
        si.uv = Point2f(0.1f, sample2);

        uint32_t flags = (uint32_t) DiscontinuityFlags::InteriorType;
        SilhouetteSample3f ss =
            primitive_silhouette_projection(viewpoint, si, flags, 0.f, active);

        Vector3f dp_dv;
        std::tie(std::ignore, dp_dv, std::ignore, std::ignore, std::ignore,
                 std::ignore, std::ignore) = partials(ss.uv, active);
        ss.pdf = dr::rcp(dr::abs(dr::dot(dp_dv, ss.silhouette_d)));

        return ss;
    }

    //! @}
    // =============================================================

    void traverse(TraversalCallback *cb) override {
        Base::traverse(cb);
        cb->put("control_point_count", m_control_point_count, ParamFlags::NonDifferentiable);
        cb->put("segment_indices",     m_indices,             ParamFlags::NonDifferentiable);
        cb->put("control_points",      m_control_points,      ParamFlags::Differentiable | ParamFlags::Discontinuous);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (keys.empty() || string::contains(keys, "control_points")) {
            recompute_bbox();
            mark_dirty();
        }
        Base::parameters_changed();
    }

    bool parameters_grad_enabled() const override {
        return dr::grad_enabled(m_control_points);
    }

    void describe(ShapeIR &g) const override {
        Base::describe(g);
        g.kind = ShapeIR::Kind::LinearCurve;
        g.cp_count = (size_t) m_control_point_count;
        g.seg_count = (size_t) dr::width(m_indices);
        g.cp_ptr  = m_control_points.data();
        g.seg_ptr = m_indices.data();
    }

    ScalarBoundingBox3f bbox() const override {
        return m_bbox;
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "LinearCurve[" << std::endl
            << "  control_point_count = " << m_control_point_count << "," << std::endl
            << "  segment_count = " << dr::width(m_indices) << "," << std::endl
            << "  " << string::indent(get_children_string()) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS(LinearCurve)

private:
    template <bool Negate, ScalarSize N>
    void advance(const char **start_, const char *end, const char (&delim)[N]) {
        const char *start = *start_;

        while (true) {
            bool is_delim = false;
            for (ScalarSize i = 0; i < N; ++i)
                if (*start == delim[i])
                    is_delim = true;
            if ((is_delim ^ Negate) || start == end)
                break;
            ++start;
        }

        *start_ = start;
    }

    void recompute_bbox() {
        m_bbox.reset();
        if (m_control_point_count == 0)
            return;

        if constexpr (dr::is_jit_v<Float>) {
            m_bbox = device_reduce_bbox<ScalarPoint3f>(
                m_control_points, m_control_point_count, 4, /* radius_offset = */ 3);
        } else {
            const InputFloat *ptr = m_control_points.data();
            for (ScalarSize i = 0; i < m_control_point_count; ++i) {
                ScalarPoint3f p(ptr[4 * i + 0], ptr[4 * i + 1], ptr[4 * i + 2]);
                ScalarFloat r(ptr[4 * i + 3]);
                m_bbox.expand(p + r * ScalarVector3f(-1, 0, 0));
                m_bbox.expand(p + r * ScalarVector3f(1, 0, 0));
                m_bbox.expand(p + r * ScalarVector3f(0, -1, 0));
                m_bbox.expand(p + r * ScalarVector3f(0, 1, 0));
                m_bbox.expand(p + r * ScalarVector3f(0, 0, -1));
                m_bbox.expand(p + r * ScalarVector3f(0, 0, 1));
            }
        }
    }

    std::tuple<Vector3f, Vector3f>
    local_frame(const Vector3f &dc_dv_normalized) const {
        // Define consistent local frame
        // (1) Consistently define a rotation axis (`v_rot`) that lies in the hemisphere defined by `guide`
        // (2) Rotate `dc_du` by 90 degrees on `v_rot` to obtain `v_rad`
        Vector3f guide = Vector3f(0, 0, 1);
        Vector3f v_rot = dr::normalize(
            guide - dc_dv_normalized * dr::dot(dc_dv_normalized, guide));
        Mask singular_mask = dr::abs(dr::dot(guide, dc_dv_normalized)) == 1.f;
        dr::masked(v_rot, singular_mask) =
            Vector3f(0, 1, 0); // non-consistent at singular points
        Vector3f v_rad = dr::cross(v_rot, dc_dv_normalized);

        return { v_rot, v_rad };
    }

    /**
     * \brief Geometry of the "round cone" (swept sphere) of segment
     * \c prim_idx at the segment-local parameter \c v_local.
     *
     * Returns the axis point \c c, the unit axis \c axis, the segment length
     * \c L, the radius and its local derivative, the consistent radial frame
     * \c (u_rot, u_rad) and the cone tilt \c (sin_a, cos_a) with
     * \c sin_a = (r0 - r1) / L. The surface point/normal at azimuth \c u are
     * \c n = cos_a * rad(u) + sin_a * axis and \c p = c + radius * n.
     */
    std::tuple<Point3f, Vector3f, Float, Float, Float, Vector3f, Vector3f,
               Float, Float>
    cone_geometry(const Float &v_local, const UInt32 &prim_idx,
                  Mask active) const {
        UInt32 idx = dr::gather<UInt32>(m_indices, prim_idx, active);
        Point4f c0 = dr::gather<Point4f>(m_control_points, idx, active),
                c1 = dr::gather<Point4f>(m_control_points, idx + 1, active);
        Point3f p0 = Point3f(c0.x(), c0.y(), c0.z()),
                p1 = Point3f(c1.x(), c1.y(), c1.z());
        Float r0 = c0.w(), r1 = c1.w();

        Vector3f seg = p1 - p0;
        Float L = dr::norm(seg);
        Vector3f axis = seg / L;
        Float sin_a = (r0 - r1) / L;
        Float cos_a = dr::safe_sqrt(1.f - sin_a * sin_a);

        Point3f c = p0 * (1.f - v_local) + p1 * v_local;
        Float radius = r0 * (1.f - v_local) + r1 * v_local;
        Float dr_dv = r1 - r0;

        auto [u_rot, u_rad] = local_frame(axis);

        return { c, axis, L, radius, dr_dv, u_rot, u_rad, sin_a, cos_a };
    }

    /**
     * \brief Position and normal partials and the second fundamental form of
     * the cone body.
     *
     * For a linear segment the centerline is straight and the normal is
     * constant along a ruling, so \c dn_dv = 0 and \c dp_dvv = 0.
     */
    std::tuple<Vector3f, Vector3f, Vector3f, Vector3f, Float, Float, Float>
    partials(Point2f uv, Mask active) const {
        Float v_global = uv.y();
        size_t segment_count = dr::width(m_indices);
        UInt32 segment_idx = dr::floor2int<UInt32>(v_global * segment_count);
        segment_idx = dr::clip(segment_idx, 0, (uint32_t) segment_count - 1);
        Float v_local = v_global * segment_count - segment_idx;

        auto [c, axis, L, radius, dr_dv, u_rot, u_rad, sin_a, cos_a] =
            cone_geometry(v_local, segment_idx, active);
        auto [sin_u, cos_u] = dr::sincos(uv.x() * dr::TwoPi<Float>);
        Vector3f rad      =  cos_u * u_rad + sin_u * u_rot,
                 rad_perp = -sin_u * u_rad + cos_u * u_rot; // d(rad)/d(2*pi*u)
        Normal3f n = cos_a * rad + sin_a * axis;

        // Partials w.r.t. the [0, 1) `u` and the local `v` (rescaled below)
        Vector3f dp_du  = radius * cos_a * rad_perp,
                 dp_dv  = axis * L + dr_dv * n,
                 dp_duu = -radius * cos_a * rad,
                 dp_duv = dr_dv * cos_a * rad_perp,
                 dp_dvv = dr::zeros<Vector3f>();
        Vector3f dn_du  = cos_a * rad_perp,
                 dn_dv  = dr::zeros<Vector3f>();

        // Rescale (u: [0, 1) -> [0, 2pi), v: local -> global)
        dp_du  *= dr::TwoPi<Float>;
        dp_duv *= dr::TwoPi<Float>;
        dp_duu *= dr::square(dr::TwoPi<Float>);
        dn_du  *= dr::TwoPi<Float>;
        ScalarFloat ratio = (ScalarFloat) dr::width(m_indices);
        dp_dv  *= ratio;
        dp_duv *= ratio;

        Float L_ff = dr::dot(n, dp_duu),
              M_ff = dr::dot(n, dp_duv),
              N_ff = dr::dot(n, dp_dvv);

        return { dp_du, dp_dv, dn_du, dn_dv, L_ff, M_ff, N_ff };
    }

private:
    ScalarBoundingBox3f m_bbox;

    ScalarSize m_control_point_count = 0;

    /// Holds the first primitive index of each curve
    mutable UInt32Storage m_curves_prim_idx;

    mutable UInt32Storage m_indices;
    mutable FloatStorage m_control_points;

    static constexpr float silhouette_offset = 5e-3f;

    MI_TRAVERSE_CB(Base, m_curves_prim_idx, m_indices, m_control_points)
};

MI_EXPORT_PLUGIN(LinearCurve)
NAMESPACE_END(mitsuba)
