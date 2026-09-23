# DeepCalib integration

`upstream/` is the checkout of `https://github.com/alexvbogdan/DeepCalib`, pinned
to `a04d8e0c4d4fdc362ebee016196eee5971506cb0`. It is ignored by Git; provision it
with `git clone https://github.com/alexvbogdan/DeepCalib backend/deep_calib/upstream`
and `git -C backend/deep_calib/upstream checkout a04d8e0c4d4fdc362ebee016196eee5971506cb0`.

## Camera workflow

1. Fetch one snapshot using the existing camera snapshot endpoint.
2. Send that exact image to `POST /api/camera/{id}/deepcalib/estimate` as
   `{image: "data:image/jpeg;base64,...", use_saved_profile: false}`.
3. SingleNet estimates focal length and spherical distortion on CUDA. The
   preview resamples the same image on CUDA and returns its profile, rectified
   JPEG and a camera-bound `preview_id`. No additional continuous video stream
   or per-frame model inference is started.
4. Click pairs on the **rectified** image and FMS map, or enter actual FMS X/Y
   coordinates in meters. At least four distinct, non-collinear, well-spread
   floor anchors are required; there is no UI limit on pairs.
5. Optionally click two ends of floor tile edges or straight segments and enter
   their real lengths. Repeat for each edge. Measurements must be on the same
   floor plane and within the anchor hull. Lengths alone cannot fix the FMS
   origin/orientation. Inconsistent lengths are rejected without changing the
   running calibration.
6. Save with `method: "deepcalib_camera_fms"`, `points_space: "rectified"`,
   `deepcalib_preview_id`, paired points, FMS frame and `length_constraints`.
   Geometry comes from the backend preview receipt, not client-modified fields.
   Receipts live for 24 hours (maximum 256 lightweight profiles) and expire on
   backend restart. Images are not retained in this cache.

Save persists the complete config in PostgreSQL `public.cameras.calibration_points`
and the matrix in `homography_matrix`, then activates it through the existing
manual-calibration transaction. Automatic robot calibration is stopped for that
camera. The saved profile, raw/rectified anchors, length constraints, residuals,
FMS coordinate frame and save ID restore at startup. A failed save does not
replace the active mapping. `use_saved_profile: true` generates a new preview
without rerunning SingleNet, allowing existing pairs to be edited safely. Send
the current `calibration_save_id` when reopening a saved profile. Legacy profiles
without a version 2 canvas must be recalibrated rather than reusing potentially
clipped points on a new preview; their active map stays unchanged meanwhile.

## Geometry and runtime

`geometry.py` implements the unified spherical model used by upstream
`undistortion/undistSphIm.m` and `undistortion/undist.py`. Preview output keeps
the snapshot resolution, with a stored output focal length/center and at most
the reference threefold field-of-view expansion. A perspective plane cannot
display rays at or behind its horizon; these points and empty image borders
are rejected. Version 2 transforms never clamp different points onto an edge.
Legacy saved profiles retain their old canvas/clipping for compatibility until
the user recalibrates, rather than silently changing an active map.

The homography consumes rectified normalized coordinates. Saved `src_points`
and `coverage_polygon` stay in the raw domain for existing consumers;
`rectified_src_points` and `rectified_coverage_polygon` describe the solver domain.
Live raw detection ground points and raw-image ruler clicks are rectified once
before applying the same homography. Rectified points bypass this step. The
measure API accepts `points_space` and optionally `calibration_save_id` to reject
measurements made against a changed calibration. Tracking never runs SingleNet
or rectifies a full video frame.

The legacy H5 topology is loaded through isolated Keras 3/Torch dependencies in
`.runtime/`. Preprocessing matches the upstream training generator: RGB in
[-1, 1], followed by `imagenet_utils.preprocess_input`'s default Caffe channel
order/mean subtraction. Model inference, resize and preview remap require CUDA;
JPEG decode/encode, small point transforms and homography fitting remain host
operations. This model is not claimed to be a TensorRT engine. It only runs on
request; errors never silently fall back to a raw homography labelled DeepCalib.

The downloaded upstream weights are intentionally not checked into source
control. Set `DEEP_CALIB_WEIGHTS` to a compatible `.h5` file or place the
provided SingleNet weights at `backend/deep_calib/weights/weights_06_5.61.h5`.
