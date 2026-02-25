# Response 02: MotionBERT Coordinates & Final Decisions

## The Normalized Coordinate "Problem" — It's Not One

After deep research, **MotionBERT's normalized output is fine for our pipeline.** Here's why:

Every single lifting model (MotionBERT, MotionAGFormer, PoseFormerV2, MixSTE, D3DP, etc.) faces the same fundamental issue: **monocular 3D pose estimation cannot recover absolute metric scale.** It's mathematically impossible without knowing the camera focal length or a reference measurement.

What lifting models give you is a **self-consistent 3D skeleton** — correct proportions, correct angles, correct relative positions. The absolute scale (mm vs normalized) is irrelevant for us because:

1. **Camera fitting absorbs the scale.** When we fit a weak perspective camera from 2D-3D correspondences, the camera matrix maps from whatever-units-the-3D-is-in to pixel coordinates. If the 3D is scaled by 1000x, the camera parameters just compensate. The 2D reprojection is identical.

2. **Heatmap optimization is scale-invariant.** Our loss function projects 3D joints → 2D pixels via the camera, then samples heatmaps. The camera handles the unit conversion.

3. **Bone length ratios are preserved.** Upper-arm-to-forearm ratio is the same whether in mm or normalized units. Our shared learnable bone lengths will converge to correct proportions.

4. **Angles are unitless.** The azimuth/elevation/theta initialization from 3D joint positions uses vector directions, not magnitudes.

So: **stick with MotionBERT.** Switching would gain nothing on coordinate systems.

---

## Alternatives Surveyed (For Reference)

If we ever wanted to swap MotionBERT, the best option is:

**MotionAGFormer** — similar accuracy (38.4 vs 37.2 mm MPJPE on H3.6M), 4x fewer parameters, 3x faster. Same normalized output convention. Pretrained weights available. But no strong reason to switch right now.

Others considered: PoseFormerV2, MixSTE, D3DP (diffusion-based), MHFormer, STCFormer. All follow the same pattern.

---

## Locked-In Decisions

Based on your responses to response01 + this research:

| Decision | Choice |
|----------|--------|
| 2D detection | YOLOv8-nano (bounding boxes) |
| 2D pose + heatmaps | ViTPose (from HuggingFace Transformers) |
| 3D lifting | MotionBERT-Lite (keep as-is) |
| Camera model | Weak perspective (4 params per frame) |
| 3D → arm init | Simple vector geometry (no IK) |
| Video recording | Not needed |
| All outputs saved to | `training_runs/<run-name>/` |
| Scope | Left arm only, single person |

---

## Existing Code Reuse Question

You said "let's do more research first" on whether to reuse `model/arm.py` and `scoring.py`. Now that the model/camera decisions are settled:

**Option A: Import from existing codebase**
- Reuse `model/arm.py` (Arm class, polar coordinates, FK)
- Reuse scoring logic (heatmap sampling via grid_sample, motion penalties)
- Adapt camera model (switch from affine/pinhole to weak perspective)
- Pro: Less code to write, already debugged
- Con: Dependency on sibling workspace

**Option B: Copy relevant pieces into a fresh single file**
- Extract the Arm class, scoring functions, etc. into the new `main.py`
- Adapt as needed for the new pipeline (ViTPose heatmaps, weak perspective camera)
- Pro: Self-contained, no cross-workspace dependencies
- Con: Code duplication

**Option C: Shared model package**
- Keep `model/` as a shared package, import from both workspaces
- Pro: Single source of truth
- Con: More project restructuring

I'd recommend **Option B** — a self-contained single file that incorporates the proven Arm + scoring logic. Clean break, no dependencies, easy to hand off. The Arm class is ~100 lines, scoring is ~80 lines. Small enough to inline.

---

## Ready to Build?

If you're happy with these decisions, I'll start writing the fresh `main.py`. The pipeline will be:

```
Video → YOLOv8 → ViTPose (heatmaps + 2D keypoints)
     → MotionBERT (3D keypoints)
     → Weak perspective camera fit (all 17 joints)
     → Arm initialization (simple geometry from left arm 3D)
     → Adam optimization (heatmap loss + motion smoothness)
     → Outputs: heatmaps, overlays, graphs, summary → training_runs/
     → VPython interactive visualization
```

Let me know if anything needs adjusting, or if I should start building.
