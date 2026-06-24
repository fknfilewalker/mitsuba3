#!/usr/bin/env python
"""
Gradient validation for the ``bsplinecurve`` shape.

Renders a single B-spline curve lit by a rectangular area light and, for every
control-point parameter (4 points x {x, y, z, radius} = 16 parameters), compares

  * the autodiff (forward-mode) gradient image produced by a projective
    integrator, against
  * a *centered* finite-difference image with eps = 0.005.

Run with the project's build + uv environment, e.g.

    PYTHONPATH=$PWD/build/python ./.venv/bin/python bsplinecurve_grad_check.py

Outputs (next to this script):
  * grad_render.png        -- the primal render (for context)
  * grad_comparison.png    -- 16 rows x (autodiff | finite diff | difference)
  * grad_scalars.csv       -- d(sum of image)/d(param): autodiff vs finite diff
"""

import os
import sys
import tempfile

# Make the script runnable directly with the venv python.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "build", "python"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

import drjit as dr
import mitsuba as mi

mi.set_variant("metal_ad_rgb")

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
EPS    = 0.005          # finite-difference step (centered)
RES    = 128            # image resolution
SPP_AD = 64            # samples for the autodiff render (forward mode is costly)
SPP_FD = 64           # samples for each finite-difference render (lower noise)
SEED   = 0

# Curve control points: [x, y, z, radius]
CONTROL_POINTS = [
    [-0.8,  0.7, 0.0, 0.1],
    [-0.5, -0.2, 0.0, 0.1],
    [ 0.5, -0.2, 0.0, 0.1],
    [ 0.8,  0.7, 0.0, 0.1],
    [ 0.9,  0.8, 0.0, 0.1],
]

# ----------------------------------------------------------------------------
# Scene construction
# ----------------------------------------------------------------------------
def write_curve_file(control_points):
    f = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False)
    f.write("".join("%f %f %f %f\n" % tuple(p) for p in control_points))
    f.close()
    return f.name


def make_scene(curve_path):
    T = mi.ScalarTransform4f
    return mi.load_dict({
        "type": "scene",
        "integrator": {
            "type": "direct_projective",
            "sppc": SPP_AD,   # continuous (interior) term
            "sppp": SPP_AD,   # primarily visible discontinuities (silhouettes)
            "sppi": SPP_AD,   # no indirect discontinuities for this test
        },
        "sensor": {
            "type": "perspective",
            "fov": 45,
            "to_world": T().look_at(origin=[0, 0, 4], target=[0, 0, 0], up=[0, 1, 0]),
            "film": {
                "type": "hdrfilm",
                "width": RES, "height": RES,
                "rfilter": {"type": "gaussian"},
                "sample_border": True,
            },
        },
        "curve": {
            "type": "bsplinecurve",
            "filename": curve_path,
            "bsdf": {"type": "diffuse", "reflectance": {"type": "rgb", "value": [0.8, 0.4, 0.2]}},
        },
        # Rectangular area light at (0, 3, 4), facing the origin. Mitsuba's
        # look_at maps the local +Z (the rectangle's normal / emission side)
        # onto (target - origin), so the light shines toward (0, 0, 0).
        "light": {
            "type": "rectangle",
            "to_world": T().look_at(origin=[0, 3, 4], target=[0, 0, 0], up=[0, 1, 0]) @ T().scale(1.5),
            "emitter": {"type": "area", "radiance": {"type": "rgb", "value": [25, 25, 25]}},
        },
    })


# ----------------------------------------------------------------------------
# Gradient evaluation
# ----------------------------------------------------------------------------
PARAM_COORD = ["x", "y", "z", "r"]
def param_label(i):
    return "P%d.%s" % (i // 4, PARAM_COORD[i % 4])


def one_hot(i, n):
    v = dr.zeros(mi.Float, n)
    dr.scatter(v, mi.Float(1.0), mi.UInt32(i))
    return v


def main(indices=None):
    curve_path = write_curve_file(CONTROL_POINTS)
    scene = make_scene(curve_path)
    os.unlink(curve_path)

    params = mi.traverse(scene)
    key = next(k for k in params.keys() if k.endswith("control_points"))
    n = dr.width(params[key])
    # `indices` selects which parameters to evaluate (default: all). Accepts a
    # single int or a list of ints.
    if indices is None:
        sel = list(range(n))
    else:
        idx = indices if isinstance(indices, (list, tuple)) else [indices]
        sel = [i for i in idx if 0 <= i < n]
    print("Differentiable parameter '%s' with %d entries; evaluating %s\n"
          % (key, n, [param_label(i) for i in sel]))

    init = mi.Float(params[key])            # pristine, detached copy
    img_to_np = lambda t: np.array(t)       # TensorXf -> (H, W, 3)

    # Primal reference render
    params[key] = mi.Float(init); params.update()
    ref = mi.render(scene, spp=SPP_FD, seed=SEED)
    mi.util.write_bitmap(os.path.join(_HERE, "grad_render.png"), ref)

    ad_imgs, fd_imgs, scal_ad, scal_fd, cosines, rel_l2 = [], [], [], [], [], []

    for i in tqdm(sel, desc="parameters", unit="param"):
        e = one_hot(i, n)

        # ---- Autodiff (forward mode): dImage / d(param_i) ----
        theta = mi.Float(0.0)
        dr.enable_grad(theta)
        params[key] = init + theta * e
        params.update()
        img = mi.render(scene, params, spp=SPP_AD, seed=SEED)
        dr.forward(theta)
        grad_ad = img_to_np(dr.grad(img))

        # ---- Centered finite difference, eps = EPS ----
        with dr.suspend_grad():
            params[key] = init + EPS * e; params.update()
            img_p = mi.render(scene, params, spp=SPP_FD, seed=SEED)
            params[key] = init - EPS * e; params.update()
            img_m = mi.render(scene, params, spp=SPP_FD, seed=SEED)
        grad_fd = (img_to_np(img_p) - img_to_np(img_m)) / (2.0 * EPS)

        a = grad_ad.sum(axis=2)   # collapse RGB for display / metrics
        f = grad_fd.sum(axis=2)
        ad_imgs.append(a); fd_imgs.append(f)
        scal_ad.append(float(a.sum()))
        scal_fd.append(float(f.sum()))
        # Image-level agreement metrics. The summed scalar d(sum)/dp cancels for
        # translational parameters (the silhouette gradient is a +/- dipole), so
        # the cosine similarity / relative L2 of the gradient *images* are the
        # meaningful checks.
        cosines.append(float((a * f).sum() /
                             (np.linalg.norm(a) * np.linalg.norm(f) + 1e-12)))
        rel_l2.append(float(np.linalg.norm(a - f) /
                            (np.linalg.norm(f) + 1e-12)))
        tqdm.write("  %-6s  cos(img)=%.3f  relL2=%.3f   d(sum): AD=%+9.1f FD=%+9.1f"
                   % (param_label(i), cosines[-1], rel_l2[-1], scal_ad[-1], scal_fd[-1]))

    params[key] = mi.Float(init); params.update()

    # ---- Comparison table ----
    with open(os.path.join(_HERE, "grad_scalars.csv"), "w") as f:
        f.write("param,cos_img,rel_l2,dsum_autodiff,dsum_finite_diff\n")
        for k, i in enumerate(sel):
            f.write("%s,%.4f,%.4f,%.6f,%.6f\n"
                    % (param_label(i), cosines[k], rel_l2[k],
                       scal_ad[k], scal_fd[k]))
    print("mean cos(img) = %.3f" % (sum(cosines) / len(cosines)))

    # ---- Image comparison grid ----
    m = len(sel)
    fig, axes = plt.subplots(m, 3, figsize=(7.5, 2.4 * m), squeeze=False)
    axes[0, 0].set_title("finite diff (eps=%.3f)" % EPS)
    axes[0, 1].set_title("autodiff (forward)")
    axes[0, 2].set_title("autodiff - finite diff")
    for k, i in enumerate(sel):
        vmax = max(np.abs(ad_imgs[k]).max(), np.abs(fd_imgs[k]).max(), 1e-8)
        for col, data in enumerate((fd_imgs[k], ad_imgs[k], ad_imgs[k] - fd_imgs[k])):
            ax = axes[k, col]
            ax.imshow(data, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
        axes[k, 0].set_ylabel(param_label(i), rotation=0, labelpad=22, va="center")
    fig.tight_layout()
    out = os.path.join(_HERE, "grad_comparison.png")
    fig.savefig(out, dpi=90)
    print("\nWrote grad_render.png, grad_comparison.png, grad_scalars.csv")


if __name__ == "__main__":
    main(indices=[10])
    # main()
