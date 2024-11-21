import warnings
from collections import OrderedDict

import numpy as np
import sophon.sail as sail
from video_utils import (
    concat_points,
    get_1d_sine_pe,
    interpolate,
    reflection_padding,
    select_closest_cond_frames,
    sigmoid,
    trunc_normal,
)

# a large negative value as a placeholder score for missing objects
NO_OBJ_SCORE = -1024.0


class SAM2VideoBase:

    def __init__(
        self,
        dev_id,
        image_encoder_path,
        image_decoder_path,
        memory_attention_path,
        memory_encoder_path,
        constant_path,
        num_mask_mem=7,
        max_obj_ptrs_in_encoder=16,
        fill_hole_area=0,
        # whether to apply non-overlapping constraints on the output object masks
        non_overlap_masks=False,
        # whether to clear non-conditioning memory of the surrounding frames (which may contain outdated information) after adding correction clicks;
        # note that this would only apply to *single-object tracking* unless `clear_non_cond_mem_for_multi_obj` is also set to True)
        clear_non_cond_mem_around_input=False,
        # whether to also clear non-conditioning memory of the surrounding frames (only effective when `clear_non_cond_mem_around_input` is True).
        clear_non_cond_mem_for_multi_obj=False,
    ):
        self.image_encoder = sail.Engine(image_encoder_path, dev_id, sail.IOMode.SYSIO)
        self.image_encoder_graph_name = self.image_encoder.get_graph_names()[0]
        self.image_decoder = sail.Engine(image_decoder_path, dev_id, sail.IOMode.SYSIO)
        self.image_decoder_graph_name = self.image_decoder.get_graph_names()[0]
        self.memory_attention = sail.Engine(
            memory_attention_path, dev_id, sail.IOMode.SYSIO
        )
        self.memory_attention_graph_name = self.memory_attention.get_graph_names()[0]
        self.memory_encoder = sail.Engine(
            memory_encoder_path, dev_id, sail.IOMode.SYSIO
        )
        self.memory_encoder_graph_name = self.memory_encoder.get_graph_names()[0]

        self.position_encoding = np.load(f"{constant_path}/pos.npz")
        self.maskmem_pos_enc = np.load(f"{constant_path}/maskmem_pos_enc.npz")

        self.num_maskmem = num_mask_mem
        self.max_obj_ptrs_in_encoder = max_obj_ptrs_in_encoder

        self.fill_hole_area = fill_hole_area
        self.non_overlap_masks = non_overlap_masks
        self.clear_non_cond_mem_around_input = clear_non_cond_mem_around_input
        self.clear_non_cond_mem_for_multi_obj = clear_non_cond_mem_for_multi_obj

    def init_state(self):
        """default state from yaml"""
        self.image_size = 1024
        self.num_feature_levels = 3
        self.hidden_dim = 256
        self.directly_add_no_mem_embed = True
        self.training = False
        self.mem_dim = 64
        self.add_tpos_enc_to_obj_ptrs = False
        self.use_obj_ptrs_in_encoder = True
        self.add_all_frames_to_correct_as_cond = False
        self.multimask_output_in_sam = True
        self.multimask_min_pt_num = 0
        self.multimask_max_pt_num = 1
        self.sam_prompt_embed_dim = self.hidden_dim
        self.backbone_stride = 16
        self.sam_image_embedding_size = self.image_size // self.backbone_stride
        self.pred_obj_scores = True
        self.use_obj_ptrs_in_encoder = True
        self.use_mlp_for_obj_ptr_proj = True
        self.proj_tpos_enc_in_obj_ptrs = False
        self.soft_no_obj_ptr = False
        self.fixed_no_obj_ptr = True
        self.non_overlap_masks_for_mem_enc = False
        self.binarize_mask_from_pts_for_mem_enc = True
        self.sigmoid_scale_for_mem_enc = 20
        self.sigmoid_bias_for_mem_enc = -10.0
        self.dynamic_multimask_via_stability = True
        self.dynamic_multimask_stability_delta = 0.05
        self.dynamic_multimask_stability_thresh = 0.98
        self.max_cond_frames_in_attn = -1
        self.memory_temporal_stride_for_eval = 1
        self.only_obj_ptrs_in_the_past_for_eval = True
        self.multimask_output_for_tracking = True
        self.use_multimask_token_for_obj_ptr = True

        # Temporal encoding of the memories
        self.maskmem_tpos_enc = trunc_normal(
            (self.num_maskmem, 1, 1, self.mem_dim), std=0.02
        )
        # a single token to indicate no memory embedding from previous frames
        self.no_mem_embed = trunc_normal((1, 1, self.hidden_dim), std=0.02)
        self.no_mem_pos_enc = trunc_normal((1, 1, self.hidden_dim), std=0.02)
        # Part 4: SAM-style prompt encoder (for both mask and point inputs)
        # and SAM-style mask decoder for the final mask output
        self.no_obj_ptr = trunc_normal((1, self.hidden_dim), std=0.02)

        """Initialize an inference state."""
        inference_state = {}
        # the original video height and width, used for resizing final output scores
        # inputs on each frame
        inference_state["point_inputs_per_obj"] = {}
        inference_state["mask_inputs_per_obj"] = {}
        # visual features on a small number of recently visited frames for quick interactions
        inference_state["cached_features"] = {}
        # values that don't change across frames (so we only need to hold one copy of them)
        inference_state["constants"] = {}
        # mapping between client-side object id and model-side object index
        inference_state["obj_id_to_idx"] = OrderedDict()
        inference_state["obj_idx_to_id"] = OrderedDict()
        inference_state["obj_ids"] = []
        # A storage to hold the model's tracking results and states on each frame
        inference_state["output_dict"] = {
            "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
        }
        # Slice (view) of each object tracking results, sharing the same memory with "output_dict"
        inference_state["output_dict_per_obj"] = {}
        # A temporary storage to hold new outputs when user interact with a frame
        # to add clicks or mask (it's merged into "output_dict" before propagation starts)
        inference_state["temp_output_dict_per_obj"] = {}
        # Frames that already holds consolidated outputs from click or mask inputs
        # (we directly use their consolidated outputs during tracking)
        inference_state["consolidated_frame_inds"] = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }
        # metadata for each tracking frame (e.g. which direction it's tracked)
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"] = {}
        # Warm up the visual backbone and cache the image feature on frame 0
        inference_state["images"] = None
        inference_state["num_frames"] = 0

        return inference_state

    def append_image(self, inference_state, image):
        inference_state["video_height"] = self.video_height
        inference_state["video_width"] = self.video_width
        inference_state["images"] = image
        # if inference_state["images"] is None:
        #     inference_state["images"] = [image]
        # else:
        #     inference_state["images"].append(image)
        inference_state["num_frames"] += 0 
        if len(inference_state["images"]) == 1:
            self._get_image_feature(inference_state, frame_idx=0)

    def _obj_id_to_idx(self, inference_state, obj_id):
        """Map client-side object id to model-side object index."""
        obj_idx = inference_state["obj_id_to_idx"].get(obj_id, None)
        if obj_idx is not None:
            return obj_idx

        # This is a new object id not sent to the server before. We only allow adding
        # new objects *before* the tracking starts.
        allow_new_object = not inference_state["tracking_has_started"]
        if allow_new_object:
            # get the next object slot
            obj_idx = len(inference_state["obj_id_to_idx"])
            inference_state["obj_id_to_idx"][obj_id] = obj_idx
            inference_state["obj_idx_to_id"][obj_idx] = obj_id
            inference_state["obj_ids"] = list(inference_state["obj_id_to_idx"])
            # set up input and output structures for this object
            inference_state["point_inputs_per_obj"][obj_idx] = {}
            inference_state["mask_inputs_per_obj"][obj_idx] = {}
            inference_state["output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            inference_state["temp_output_dict_per_obj"][obj_idx] = {
                "cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
                "non_cond_frame_outputs": {},  # dict containing {frame_idx: <out>}
            }
            return obj_idx
        else:
            raise RuntimeError(
                f"Cannot add new object id {obj_id} after tracking starts. "
                f"All existing object ids: {inference_state['obj_ids']}. "
                f"Please call 'reset_state' to restart from scratch."
            )

    def _obj_idx_to_id(self, inference_state, obj_idx):
        """Map model-side object index to client-side object id."""
        return inference_state["obj_idx_to_id"][obj_idx]

    def _get_obj_num(self, inference_state):
        """Get the total number of unique object ids received so far in this session."""
        return len(inference_state["obj_idx_to_id"])

    def add_new_points_or_box(
        self,
        inference_state,
        frame_idx,
        obj_id,
        points=None,
        labels=None,
        clear_old_points=True,
        normalize_coords=True,
        box=None,
    ):
        """Add new points to a frame."""
        obj_idx = self._obj_id_to_idx(inference_state, obj_id)
        point_inputs_per_frame = inference_state["point_inputs_per_obj"][obj_idx]
        mask_inputs_per_frame = inference_state["mask_inputs_per_obj"][obj_idx]

        if (points is not None) != (labels is not None):
            raise ValueError("points and labels must be provided together")
        if points is None and box is None:
            raise ValueError("at least one of points or box must be provided as input")

        if points is None or len(points) == 0:
            points = np.zeros((0, 2), dtype=np.float32)
        if labels is None or len(labels) == 0:
            labels = np.zeros((0), dtype=np.int32)
        if points.ndim == 2:
            points = np.expand_dims(points, axis=0)  # add batch dimension
        if labels.ndim == 1:
            labels = np.expand_dims(labels, axis=0)  # add batch dimension

        # If `box` is provided, we add it as the first two points with labels 2 and 3
        # along with the user-provided points (consistent with how SAM 2 is trained).
        if box is not None:
            if not clear_old_points:
                raise ValueError(
                    "cannot add box without clearing old points, since "
                    "box prompt must be provided before any point prompt "
                    "(please use clear_old_points=True instead)"
                )
            if inference_state["tracking_has_started"]:
                warnings.warn(
                    "You are adding a box after tracking starts. SAM 2 may not always be "
                    "able to incorporate a box prompt for *refinement*. If you intend to "
                    "use box prompt as an *initial* input before tracking, please call "
                    "'reset_state' on the inference state to restart from scratch.",
                    category=UserWarning,
                    stacklevel=2,
                )
            box_coords = box.reshape(1, 2, 2)
            box_labels = np.array([2, 3], dtype=np.int32)
            box_labels = box_labels.reshape(1, 2)
            points = np.concatenate([box_coords, points], axis=1)
            labels = np.concatenate([box_labels, labels], axis=1)

        if normalize_coords:
            video_H = inference_state["video_height"]
            video_W = inference_state["video_width"]
            points = points / np.array([video_W, video_H])
        # scale the (normalized) coordinates by the model's internal image size
        points = points * self.image_size

        if not clear_old_points:
            point_inputs = point_inputs_per_frame.get(frame_idx, None)
        else:
            point_inputs = None
        point_inputs = concat_points(point_inputs, points, labels)

        point_inputs_per_frame[frame_idx] = point_inputs
        mask_inputs_per_frame.pop(frame_idx, None)
        # If this frame hasn't been tracked before, we treat it as an initial conditioning
        # frame, meaning that the inputs points are to generate segments on this frame without
        # using any memory from other frames, like in SAM. Otherwise (if it has been tracked),
        # the input points will be used to correct the already tracked masks.
        is_init_cond_frame = frame_idx not in inference_state["frames_already_tracked"]
        # whether to track in reverse time order
        if is_init_cond_frame:
            reverse = False
        else:
            reverse = inference_state["frames_already_tracked"][frame_idx]["reverse"]
        
        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
        obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
        # Add a frame to conditioning output if it's an initial conditioning frame or
        # if the model sees all frames receiving clicks/mask as conditioning frames.
        is_cond = is_init_cond_frame or self.add_all_frames_to_correct_as_cond
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"

        # Get any previously predicted mask logits on this object and feed it along with
        # the new clicks into the SAM mask decoder.
        prev_sam_mask_logits = None
        # lookup temporary output dict first, which contains the most recent output
        # (if not found, then lookup conditioning and non-conditioning frame output)
        prev_out = obj_temp_output_dict[storage_key].get(frame_idx)
        if prev_out is None:
            prev_out = obj_output_dict["cond_frame_outputs"].get(frame_idx)
            if prev_out is None:
                prev_out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx)

        if prev_out is not None and prev_out["pred_masks"] is not None:
            prev_sam_mask_logits = prev_out["pred_masks"]
            # Clamp the scale of prev_sam_mask_logits to avoid rare numerical issues.
            prev_sam_mask_logits = np.clip(prev_sam_mask_logits, -32.0, 32.0)
        current_out, _ = self._run_single_frame_inference(
            inference_state=inference_state,
            output_dict=obj_output_dict,  # run on the slice of a single object
            frame_idx=frame_idx,
            batch_size=1,  # run on the slice of a single object
            is_init_cond_frame=is_init_cond_frame,
            point_inputs=point_inputs,
            mask_inputs=None,
            reverse=reverse,
            # Skip the memory encoder when adding clicks or mask. We execute the memory encoder
            # at the beginning of `propagate_in_video` (after user finalize their clicks). This
            # allows us to enforce non-overlapping constraints on all objects before encoding
            # them into memory.
            run_mem_encoder=False,
            prev_sam_mask_logits=prev_sam_mask_logits,
        )
        # Add the output to the output dict (to be used as future memory)
        obj_temp_output_dict[storage_key][frame_idx] = current_out

        # Resize the output mask to the original video resolution
        obj_ids = inference_state["obj_ids"]
        consolidated_out = self._consolidate_temp_output_across_obj(
            inference_state,
            frame_idx,
            is_cond=is_cond,
            run_mem_encoder=False,
            consolidate_at_video_res=True,
        )
        _, video_res_masks = self._get_orig_video_res_output(
            inference_state, consolidated_out["pred_masks_video_res"]
        )
        return frame_idx, obj_ids, video_res_masks

    def _get_orig_video_res_output(self, inference_state, any_res_masks):
        """
        Resize the object scores to the original video resolution (video_res_masks)
        and apply non-overlapping constraints for final output.
        """
        video_H = inference_state["video_height"]
        video_W = inference_state["video_width"]
        if any_res_masks.shape[-2:] == (video_H, video_W):
            video_res_masks = any_res_masks
        else:
            video_res_masks = interpolate(any_res_masks, (video_H, video_W))

        if self.non_overlap_masks:
            video_res_masks = self._apply_non_overlapping_constraints(video_res_masks)
        return any_res_masks, video_res_masks

    def _consolidate_temp_output_across_obj(
        self,
        inference_state,
        frame_idx,
        is_cond,
        run_mem_encoder,
        consolidate_at_video_res=False,
        image_encoder=None,
        image_decoder=None,
        memory_attention=None,
        memory_encoder=None,
    ):
        """
        Consolidate the per-object temporary outputs in `temp_output_dict_per_obj` on
        a frame into a single output for all objects, including
        1) fill any missing objects either from `output_dict_per_obj` (if they exist in
           `output_dict_per_obj` for this frame) or leave them as placeholder values
           (if they don't exist in `output_dict_per_obj` for this frame);
        2) if specified, rerun memory encoder after apply non-overlapping constraints
           on the object scores.
        """
        batch_size = self._get_obj_num(inference_state)
        storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
        # Optionally, we allow consolidating the temporary outputs at the original
        # video resolution (to provide a better editing experience for mask prompts).
        if consolidate_at_video_res:
            assert not run_mem_encoder, "memory encoder cannot run at video resolution"
            consolidated_H = inference_state["video_height"]
            consolidated_W = inference_state["video_width"]
            consolidated_mask_key = "pred_masks_video_res"
        else:
            consolidated_H = consolidated_W = self.image_size // 4
            consolidated_mask_key = "pred_masks"

        # Initialize `consolidated_out`. Its "maskmem_features" and "maskmem_pos_enc"
        # will be added when rerunning the memory encoder after applying non-overlapping
        # constraints to object scores. Its "pred_masks" are prefilled with a large
        # negative value (NO_OBJ_SCORE) to represent missing objects.
        consolidated_out = {
            "maskmem_features": None,
            "maskmem_pos_enc": None,
            consolidated_mask_key: np.full(
                shape=(batch_size, 1, consolidated_H, consolidated_W),
                fill_value=NO_OBJ_SCORE,
                dtype=np.float32,
            ),
            "obj_ptr": np.full(
                shape=(batch_size, self.hidden_dim),
                fill_value=NO_OBJ_SCORE,
                dtype=np.float32,
            ),
        }
        empty_mask_ptr = None
        for obj_idx in range(batch_size):
            obj_temp_output_dict = inference_state["temp_output_dict_per_obj"][obj_idx]
            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
            out = obj_temp_output_dict[storage_key].get(frame_idx, None)
            # If the object doesn't appear in "temp_output_dict_per_obj" on this frame,
            # we fall back and look up its previous output in "output_dict_per_obj".
            # We look up both "cond_frame_outputs" and "non_cond_frame_outputs" in
            # "output_dict_per_obj" to find a previous output for this object.

            if out is None:
                out = obj_output_dict["cond_frame_outputs"].get(frame_idx, None)
            if out is None:
                out = obj_output_dict["non_cond_frame_outputs"].get(frame_idx, None)
            # If the object doesn't appear in "output_dict_per_obj" either, we skip it
            # and leave its mask scores to the default scores (i.e. the NO_OBJ_SCORE
            # placeholder above) and set its object pointer to be a dummy pointer.
            if out is None:
                # Fill in dummy object pointers for those objects without any inputs or
                # tracking outcomes on this frame (only do it under `run_mem_encoder=True`,
                # i.e. when we need to build the memory for tracking).
                if run_mem_encoder:
                    if empty_mask_ptr is None:
                        empty_mask_ptr = self._get_empty_mask_ptr(
                            inference_state,
                            frame_idx,
                            image_encoder,
                            image_decoder,
                            memory_attention,
                            memory_encoder,
                        )
                    # fill object pointer with a dummy pointer (based on an empty mask)
                    consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = empty_mask_ptr

            obj_mask = out["pred_masks"]
            consolidated_pred_masks = consolidated_out[consolidated_mask_key]

            if obj_mask.shape[-2:] == consolidated_pred_masks.shape[-2:]:
                consolidated_pred_masks[obj_idx : obj_idx + 1] = obj_mask
            else:
                # Resize first if temporary object mask has a different resolution
                resized_obj_mask = interpolate(
                    obj_mask, consolidated_pred_masks.shape[-2:]
                )

                consolidated_pred_masks[obj_idx : obj_idx + 1] = resized_obj_mask
            consolidated_out["obj_ptr"][obj_idx : obj_idx + 1] = out["obj_ptr"]

        # Optionally, apply non-overlapping constraints on the consolidated scores
        # and rerun the memory encoder
        if run_mem_encoder:
            high_res_masks = interpolate(
                consolidated_out["pred_masks"], (self.image_size, self.image_size)
            )

            if self.non_overlap_masks_for_mem_enc:
                high_res_masks = self._apply_non_overlapping_constraints(high_res_masks)
            maskmem_features, maskmem_pos_enc = self._run_memory_encoder(
                inference_state=inference_state,
                frame_idx=frame_idx,
                batch_size=batch_size,
                high_res_masks=high_res_masks,
            )
            consolidated_out["maskmem_features"] = maskmem_features
            consolidated_out["maskmem_pos_enc"] = maskmem_pos_enc

        return consolidated_out

    def propagate_in_video_preflight(self, inference_state):
        """Prepare inference_state and consolidate temporary outputs before tracking."""
        # Tracking has started and we don't allow adding new objects until session is reset.
        inference_state["tracking_has_started"] = True
        batch_size = self._get_obj_num(inference_state)

        # Consolidate per-object temporary outputs in "temp_output_dict_per_obj" and
        # add them into "output_dict".
        temp_output_dict_per_obj = inference_state["temp_output_dict_per_obj"]
        output_dict = inference_state["output_dict"]
        # "consolidated_frame_inds" contains indices of those frames where consolidated
        # temporary outputs have been added (either in this call or any previous calls
        # to `propagate_in_video_preflight`).
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        for is_cond in [False, True]:
            # Separately consolidate conditioning and non-conditioning temp outputs
            storage_key = "cond_frame_outputs" if is_cond else "non_cond_frame_outputs"
            # Find all the frames that contain temporary outputs for any objects
            # (these should be the frames that have just received clicks for mask inputs
            # via `add_new_points_or_box` or `add_new_mask`)
            temp_frame_inds = set()
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                temp_frame_inds.update(obj_temp_output_dict[storage_key].keys())
            consolidated_frame_inds[storage_key].update(temp_frame_inds)
            # consolidate the temporary output across all objects on this frame
            for frame_idx in temp_frame_inds:
                consolidated_out = self._consolidate_temp_output_across_obj(
                    inference_state, frame_idx, is_cond=is_cond, run_mem_encoder=True
                )
                # merge them into "output_dict" and also create per-object slices
                output_dict[storage_key][frame_idx] = consolidated_out
                self._add_output_per_object(
                    inference_state, frame_idx, consolidated_out, storage_key
                )
                clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
                    self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
                )
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)

            # clear temporary outputs in `temp_output_dict_per_obj`
            for obj_temp_output_dict in temp_output_dict_per_obj.values():
                obj_temp_output_dict[storage_key].clear()

        # edge case: if an output is added to "cond_frame_outputs", we remove any prior
        # output on the same frame in "non_cond_frame_outputs"
        for frame_idx in output_dict["cond_frame_outputs"]:
            output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for obj_output_dict in inference_state["output_dict_per_obj"].values():
            for frame_idx in obj_output_dict["cond_frame_outputs"]:
                obj_output_dict["non_cond_frame_outputs"].pop(frame_idx, None)
        for frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
            assert frame_idx in output_dict["cond_frame_outputs"]
            consolidated_frame_inds["non_cond_frame_outputs"].discard(frame_idx)

        # Make sure that the frame indices in "consolidated_frame_inds" are exactly those frames
        # with either points or mask inputs (which should be true under a correct workflow).
        all_consolidated_frame_inds = (
            consolidated_frame_inds["cond_frame_outputs"]
            | consolidated_frame_inds["non_cond_frame_outputs"]
        )
        input_frames_inds = set()
        for point_inputs_per_frame in inference_state["point_inputs_per_obj"].values():
            input_frames_inds.update(point_inputs_per_frame.keys())
        for mask_inputs_per_frame in inference_state["mask_inputs_per_obj"].values():
            input_frames_inds.update(mask_inputs_per_frame.keys())
        assert all_consolidated_frame_inds == input_frames_inds

    def propagate_in_video(self, inference_state, frame_idx=0):
        """Propagate the input points across frames to track in the entire video."""

        output_dict = inference_state["output_dict"]
        consolidated_frame_inds = inference_state["consolidated_frame_inds"]
        obj_ids = inference_state["obj_ids"]
        num_frames = inference_state["num_frames"]
        batch_size = self._get_obj_num(inference_state)
        if len(output_dict["cond_frame_outputs"]) == 0:
            raise RuntimeError("No points are provided; please add points first")
        clear_non_cond_mem = self.clear_non_cond_mem_around_input and (
            self.clear_non_cond_mem_for_multi_obj or batch_size <= 1
        )

        # set start index, end index, and processing order

        # default: start from the earliest frame with input points
        start_frame_idx = min(output_dict["cond_frame_outputs"])

        # default: track all the frames in the video
        max_frame_num_to_track = num_frames

        reverse = False

        end_frame_idx = min(start_frame_idx + max_frame_num_to_track, num_frames - 1)
        processing_order = range(start_frame_idx, end_frame_idx + 1)

        if True:  # for frame_idx in tqdm(processing_order, desc="propagate in video"):
            # We skip those frames already in consolidated outputs (these are frames
            # that received input clicks or mask). Note that we cannot directly run
            # batched forward on them via `_run_single_frame_inference` because the
            # number of clicks on each object might be different.
            if frame_idx in consolidated_frame_inds["cond_frame_outputs"]:
                storage_key = "cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
                if clear_non_cond_mem:
                    # clear non-conditioning memory of the surrounding frames
                    self._clear_non_cond_mem_around_input(inference_state, frame_idx)
            elif frame_idx in consolidated_frame_inds["non_cond_frame_outputs"]:
                storage_key = "non_cond_frame_outputs"
                current_out = output_dict[storage_key][frame_idx]
                pred_masks = current_out["pred_masks"]
            else:
                storage_key = "non_cond_frame_outputs"
                current_out, pred_masks = self._run_single_frame_inference(
                    inference_state=inference_state,
                    output_dict=output_dict,
                    frame_idx=frame_idx,
                    batch_size=batch_size,
                    is_init_cond_frame=False,
                    point_inputs=None,
                    mask_inputs=None,
                    reverse=reverse,
                    run_mem_encoder=True,
                )
                output_dict[storage_key][frame_idx] = current_out

            if len(output_dict[storage_key]) > 15:
                del_idx = min(output_dict[storage_key].keys())
                del output_dict[storage_key][del_idx]
            # Create slices of per-object outputs for subsequent interaction with each
            # individual object after tracking.
            self._add_output_per_object(
                inference_state, frame_idx, current_out, storage_key
            )
            inference_state["frames_already_tracked"][frame_idx] = {"reverse": reverse}

            # Resize the output mask to the original video resolution (we directly use
            # the mask scores on GPU for output to avoid any CPU conversion in between)
            _, video_res_masks = self._get_orig_video_res_output(
                inference_state, pred_masks
            )
            return frame_idx, obj_ids, video_res_masks

    def _add_output_per_object(
        self, inference_state, frame_idx, current_out, storage_key
    ):
        """
        Split a multi-object output into per-object output slices and add them into
        `output_dict_per_obj`. The resulting slices share the same tensor storage.
        """
        maskmem_features = current_out["maskmem_features"]
        # assert maskmem_features is None or isinstance(maskmem_features, np.array)

        maskmem_pos_enc = current_out["maskmem_pos_enc"]
        # assert maskmem_pos_enc is None or isinstance(maskmem_pos_enc, list)

        output_dict_per_obj = inference_state["output_dict_per_obj"]
        for obj_idx, obj_output_dict in output_dict_per_obj.items():
            obj_slice = slice(obj_idx, obj_idx + 1)
            obj_out = {
                "maskmem_features": None,
                "maskmem_pos_enc": None,
                "pred_masks": current_out["pred_masks"][obj_slice],
                "obj_ptr": current_out["obj_ptr"][obj_slice],
            }
            if maskmem_features is not None:
                obj_out["maskmem_features"] = maskmem_features[obj_slice]
            if maskmem_pos_enc is not None:
                obj_out["maskmem_pos_enc"] = [x[obj_slice] for x in maskmem_pos_enc]
            obj_output_dict[storage_key][frame_idx] = obj_out

            # 维护一个固定长度的数据窗口
            if len(obj_output_dict[storage_key]) > 15:
                # 找到并删除最旧的数据帧
                oldest_frame = min(obj_output_dict[storage_key].keys())
                del obj_output_dict[storage_key][oldest_frame]

    def reset_state(self, inference_state):
        """Remove all input points or mask in all frames throughout the video."""
        self._reset_tracking_results(inference_state)
        # Remove all object ids
        inference_state["obj_id_to_idx"].clear()
        inference_state["obj_idx_to_id"].clear()
        inference_state["obj_ids"].clear()
        inference_state["point_inputs_per_obj"].clear()
        inference_state["mask_inputs_per_obj"].clear()
        inference_state["output_dict_per_obj"].clear()
        inference_state["temp_output_dict_per_obj"].clear()

    def _reset_tracking_results(self, inference_state):
        """Reset all tracking inputs and results across the videos."""
        for v in inference_state["point_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["mask_inputs_per_obj"].values():
            v.clear()
        for v in inference_state["output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        for v in inference_state["temp_output_dict_per_obj"].values():
            v["cond_frame_outputs"].clear()
            v["non_cond_frame_outputs"].clear()
        inference_state["output_dict"]["cond_frame_outputs"].clear()
        inference_state["output_dict"]["non_cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["cond_frame_outputs"].clear()
        inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].clear()
        inference_state["tracking_has_started"] = False
        inference_state["frames_already_tracked"].clear()

    def _get_image_feature(self, inference_state, frame_idx, batch_size=1):
        """Compute the image features on a given frame."""
        # Look up in the cache first
        image = inference_state['images']
        # if backbone_out is None:
            # Cache miss -- we will run inference on a single image
        image = np.expand_dims(image, axis=0).astype(
            np.float32
        )

        outputs = self.image_encoder.process(
            self.image_encoder_graph_name, {"image": image}
        )

        backbone_fpn = [output for output in outputs.values()]

        vision_pos_enc = [pos for pos in self.position_encoding.values()]

        backbone_out = {
            "vision_features": backbone_fpn[0],
            "vision_pos_enc": vision_pos_enc,
            "backbone_fpn": backbone_fpn,
        }

        inference_state["cached_features"] = {frame_idx: (image, backbone_out)}

        # expand the features to have the same dimension as the number of objects
        expanded_image = np.repeat(image[np.newaxis, ...], batch_size, axis=0)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        # batch_size 默认为1
        expanded_image = np.repeat(image[np.newaxis, ...], batch_size, axis=0)
        expanded_backbone_out = {
            "backbone_fpn": backbone_out["backbone_fpn"].copy(),
            "vision_pos_enc": backbone_out["vision_pos_enc"].copy(),
        }
        for i, feat in enumerate(expanded_backbone_out["backbone_fpn"]):
            new_shape = (batch_size, feat.shape[1], feat.shape[2], feat.shape[3])
            expanded_backbone_out["backbone_fpn"][i] = np.broadcast_to(feat, new_shape)

        for i, pos in enumerate(expanded_backbone_out["vision_pos_enc"]):
            new_shape = (batch_size, pos.shape[1], pos.shape[2], pos.shape[3])
            pos = np.broadcast_to(pos, new_shape)
            # pos = pos.expand(batch_size, -1, -1, -1)
            expanded_backbone_out["vision_pos_enc"][i] = pos

        features = self._prepare_backbone_features(expanded_backbone_out)
        features = (expanded_image,) + features
        return features

    def _run_single_frame_inference(
        self,
        inference_state,
        output_dict,
        frame_idx,
        batch_size,
        is_init_cond_frame,
        point_inputs,
        mask_inputs,
        reverse,
        run_mem_encoder,
        prev_sam_mask_logits=None,
    ):
        """Run tracking on a single frame based on current inputs and previous memory."""
        # Retrieve correct image features

        (
            _,
            _,
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
        ) = self._get_image_feature(inference_state, frame_idx)

        # point and mask should not appear as input simultaneously on the same frame
        assert point_inputs is None or mask_inputs is None
        frame_size = np.array(
            [inference_state["video_height"], inference_state["video_width"]],
            dtype=np.int64,
        )
        # if point_inputs is not None:
        #     point_inputs = inference_state["point_inputs_per_obj"][frame_idx][0]
        current_out = self.track_step(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats,
            current_vision_pos_embeds=current_vision_pos_embeds,
            feat_sizes=feat_sizes,
            point_inputs=point_inputs,
            mask_inputs=mask_inputs,
            output_dict=output_dict,
            num_frames=inference_state["num_frames"],
            track_in_reverse=reverse,
            frame_size=frame_size,
            run_mem_encoder=run_mem_encoder,
            pix_feat=inference_state["cached_features"][frame_idx][1]["backbone_fpn"][
                -1
            ],
        )

        # optionally offload the output to CPU memory to save GPU space
        maskmem_features = current_out["maskmem_features"]
        pred_masks_gpu = current_out["pred_masks"]
        pred_masks = pred_masks_gpu
        # "maskmem_pos_enc" is the same across frames, so we only need to store one copy of it
        maskmem_pos_enc = self._get_maskmem_pos_enc(inference_state, current_out)
        # mem
        # object pointer is a small tensor, so we always keep it on GPU memory for fast access
        obj_ptr = current_out["obj_ptr"]
        # make a compact version of this frame's output to reduce the state size
        compact_current_out = {
            "maskmem_features": maskmem_features,
            "maskmem_pos_enc": maskmem_pos_enc,
            "pred_masks": pred_masks,
            "obj_ptr": obj_ptr,
        }
        return compact_current_out, pred_masks_gpu

    def _run_memory_encoder(
        self, inference_state, frame_idx, batch_size, high_res_masks
    ):
        """
        Run the memory encoder on `high_res_masks`. This is usually after applying
        non-overlapping constraints to object scores. Since their scores changed, their
        memory also need to be computed again with the memory encoder.
        """
        # Retrieve correct image features
        features = self._get_image_feature(inference_state, frame_idx)

        pix_feat = inference_state["cached_features"][frame_idx][1]["backbone_fpn"][-1]

        outputs = self.memory_encoder.process(
            self.memory_encoder_graph_name,
            {"pix_feat": pix_feat, "mask_for_mem": high_res_masks},
        )
        maskmem_features = outputs["maskmem_features_Conv_f32"]
        maskmem_pos_enc = self.maskmem_pos_enc["maskmem_pos_enc"]

        maskmem_pos_enc = self._get_maskmem_pos_enc(
            inference_state, {"maskmem_pos_enc": [maskmem_pos_enc]}
        )
        return maskmem_features, maskmem_pos_enc

    def _get_maskmem_pos_enc(self, inference_state, current_out):
        """
        `maskmem_pos_enc` is the same across frames and objects, so we cache it as
        a constant in the inference session to reduce session storage size.
        """
        model_constants = inference_state["constants"]
        # "out_maskmem_pos_enc" should be either a list of tensors or None
        out_maskmem_pos_enc = current_out["maskmem_pos_enc"]
        if out_maskmem_pos_enc is not None:
            if "maskmem_pos_enc" not in model_constants:
                assert isinstance(out_maskmem_pos_enc, list)
                # only take the slice for one object, since it's same across objects
                maskmem_pos_enc = [x[0:1].copy() for x in out_maskmem_pos_enc]
                model_constants["maskmem_pos_enc"] = maskmem_pos_enc
            else:
                maskmem_pos_enc = model_constants["maskmem_pos_enc"]
            # expand the cached maskmem_pos_enc to the actual batch size
            batch_size = out_maskmem_pos_enc[0].shape[0]

            expanded_maskmem_pos_enc = [
                np.broadcast_to(x, (batch_size, x.shape[1], x.shape[2], x.shape[3]))
                for x in maskmem_pos_enc
            ]

        else:
            expanded_maskmem_pos_enc = None
        return expanded_maskmem_pos_enc

    def _clear_non_cond_mem_around_input(self, inference_state, frame_idx):
        """
        Remove the non-conditioning memory around the input frame. When users provide
        correction clicks, the surrounding frames' non-conditioning memories can still
        contain outdated object appearance information and could confuse the model.

        This method clears those non-conditioning memories surrounding the interacted
        frame to avoid giving the model both old and new information about the object.
        """
        r = self.memory_temporal_stride_for_eval
        frame_idx_begin = frame_idx - r * self.num_maskmem
        frame_idx_end = frame_idx + r * self.num_maskmem
        output_dict = inference_state["output_dict"]
        non_cond_frame_outputs = output_dict["non_cond_frame_outputs"]
        for t in range(frame_idx_begin, frame_idx_end + 1):
            non_cond_frame_outputs.pop(t, None)
            for obj_output_dict in inference_state["output_dict_per_obj"].values():
                obj_output_dict["non_cond_frame_outputs"].pop(t, None)

    # sam2_base.py
    def track_step(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        point_inputs,
        mask_inputs,
        output_dict,
        num_frames,
        track_in_reverse=False,
        run_mem_encoder=True,
        pix_feat=None,
        frame_size=None,
        # The previously predicted SAM mask logits (which can be fed together with new clicks in demo).
    ):
        current_out = {"point_inputs": point_inputs, "mask_inputs": mask_inputs}
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:

            high_res_feature_0, high_res_feature_1 = [
                np.reshape(np.transpose(x, (1, 2, 0)), (x.shape[1], x.shape[2], *s))
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_feature_0, high_res_feature_1 = None, None

        # fused the visual feature with previous memory features in the memory bank
        pix_feat_with_mem = self._prepare_memory_conditioned_features(
            frame_idx=frame_idx,
            is_init_cond_frame=is_init_cond_frame,
            current_vision_feats=current_vision_feats[-1:],
            current_vision_pos_embeds=current_vision_pos_embeds[-1:],
            feat_sizes=feat_sizes[-1:],
            output_dict=output_dict,
            num_frames=num_frames,
            track_in_reverse=track_in_reverse,
        )
        # High-resolution feature maps for the SAM head, reshape (HW)BC => BCHW
        if len(current_vision_feats) > 1:

            high_res_features = [
                np.reshape(np.transpose(x, (1, 2, 0)), (x.shape[1], x.shape[2], *s))
                for x, s in zip(current_vision_feats[:-1], feat_sizes[:-1])
            ]
        else:
            high_res_features = None

        if not point_inputs:
            sam_point_coords = np.zeros((1, 2, 2), dtype=np.float32)
            sam_point_labels = -np.ones((1, 2), dtype=np.float32)
        else:
            sam_point_coords = point_inputs["point_coords"].astype(np.float32)
            sam_point_labels = point_inputs["point_labels"].astype(np.float32)
            assert sam_point_coords.shape[0] == 1 and sam_point_labels.shape[0] == 1

        sam_outputs = self.image_decoder.process(
            self.image_decoder_graph_name,
            {
                "point_coords": sam_point_coords,
                "point_labels": sam_point_labels,
                "image_embed": pix_feat_with_mem,
                "high_res_feats_0": high_res_feature_0,
                "high_res_feats_1": high_res_feature_1,
            },
        )

        obj_ptr = sam_outputs["obj_ptr_Add_f32"]
        mask_for_mem = sam_outputs["mask_for_mem_Add_f32"]
        video_res_mask = sam_outputs["low_res_masks_Unsqueeze_f32"]

        current_out["pred_masks"] = video_res_mask
        current_out["pred_masks_high_res"] = mask_for_mem
        current_out["obj_ptr"] = obj_ptr

        # Finally run the memory encoder on the predicted mask to encode
        # it into a new memory feature (that can be used in future frames)
        if run_mem_encoder and self.num_maskmem > 0:
            maskmem_features = self.memory_encoder.process(
                self.memory_encoder_graph_name,
                {"pix_feat": pix_feat, "mask_for_mem": mask_for_mem},
            )
            maskmem_pos_enc = self.maskmem_pos_enc["maskmem_pos_enc"]
            current_out["maskmem_features"] = maskmem_features[
                "maskmem_features_Conv_f32"
            ]
            current_out["maskmem_pos_enc"] = [maskmem_pos_enc]
        else:
            current_out["maskmem_features"] = None
            current_out["maskmem_pos_enc"] = None

        return current_out

    def _prepare_memory_conditioned_features(
        self,
        frame_idx,
        is_init_cond_frame,
        current_vision_feats,
        current_vision_pos_embeds,
        feat_sizes,
        output_dict,
        num_frames,
        track_in_reverse=False,  # tracking in reverse time order (for demo usage)
    ):
        """Fuse the current frame's visual feature map with previous memory."""
        B = current_vision_feats[-1].shape[1]  # batch size on this frame
        C = self.hidden_dim
        H, W = feat_sizes[-1]  # top-level (lowest-resolution) feature size
        # The case of `self.num_maskmem == 0` below is primarily used for reproducing SAM on images.
        # In this case, we skip the fusion with any memory.
        # current_vision_feats = np.transpose(current_vision_feats[-1], (1, 2, 0)).reshape((B, C, H, W))
        if self.num_maskmem == 0:  # Disable memory and skip fusion
            pix_feat = np.transpose(current_vision_feats[-1], (1, 2, 0)).reshape(
                (B, C, H, W)
            )
            return pix_feat

        num_obj_ptr_tokens = 0
        # Step 1: condition the visual features of the current frame on previous memories
        if not is_init_cond_frame:
            # Retrieve the memories encoded with the maskmem backbone
            to_cat_memory, to_cat_memory_pos_embed = [], []
            # Add conditioning frames's output first (all cond frames have t_pos=0 for
            # when getting temporal positional embedding below)
            assert len(output_dict["cond_frame_outputs"]) > 0
            # Select a maximum number of temporally closest cond frames for cross attention
            cond_outputs = output_dict["cond_frame_outputs"]
            selected_cond_outputs, unselected_cond_outputs = select_closest_cond_frames(
                frame_idx, cond_outputs, self.max_cond_frames_in_attn
            )
            t_pos_and_prevs = [(0, out) for out in selected_cond_outputs.values()]
            # Add last (self.num_maskmem - 1) frames before current frame for non-conditioning memory
            # the earliest one has t_pos=1 and the latest one has t_pos=self.num_maskmem-1
            # We also allow taking the memory frame non-consecutively (with r>1), in which case
            # we take (self.num_maskmem - 2) frames among every r-th frames plus the last frame.
            r = self.memory_temporal_stride_for_eval
            for t_pos in range(1, self.num_maskmem):
                t_rel = self.num_maskmem - t_pos  # how many frames before current frame
                if t_rel == 1:
                    # for t_rel == 1, we take the last frame (regardless of r)
                    if not track_in_reverse:
                        # the frame immediately before this frame (i.e. frame_idx - 1)
                        prev_frame_idx = frame_idx - t_rel
                    else:
                        # the frame immediately after this frame (i.e. frame_idx + 1)
                        prev_frame_idx = frame_idx + t_rel
                else:
                    # for t_rel >= 2, we take the memory frame from every r-th frames
                    if not track_in_reverse:
                        # first find the nearest frame among every r-th frames before this frame
                        # for r=1, this would be (frame_idx - 2)
                        prev_frame_idx = ((frame_idx - 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx - (t_rel - 2) * r
                    else:
                        # first find the nearest frame among every r-th frames after this frame
                        # for r=1, this would be (frame_idx + 2)
                        prev_frame_idx = -(-(frame_idx + 2) // r) * r
                        # then seek further among every r-th frames
                        prev_frame_idx = prev_frame_idx + (t_rel - 2) * r
                out = output_dict["non_cond_frame_outputs"].get(prev_frame_idx, None)
                if out is None:
                    # If an unselected conditioning frame is among the last (self.num_maskmem - 1)
                    # frames, we still attend to it as if it's a non-conditioning frame.
                    out = unselected_cond_outputs.get(prev_frame_idx, None)
                t_pos_and_prevs.append((t_pos, out))

            for t_pos, prev in t_pos_and_prevs:
                if prev is None:
                    continue  # skip padding frames
                # "maskmem_features" might have been offloaded to CPU in demo use cases,
                # so we load it back to GPU (it's a no-op if it's already on GPU).
                feats = prev["maskmem_features"]
                # to_cat_memory.append(feats.flatten(2).permute(2, 0, 1))
                to_cat_memory.append(
                    np.transpose(
                        np.reshape(
                            feats,
                            (
                                feats.shape[0],
                                feats.shape[1],
                                feats.shape[2] * feats.shape[3],
                            ),
                        ),
                        (2, 0, 1),
                    )
                )
                # Spatial positional encoding (it might have been offloaded to CPU in eval)
                maskmem_enc = prev["maskmem_pos_enc"][-1]
                # maskmem_enc = maskmem_enc.flatten(2).permute(2, 0, 1)
                maskmem_enc = np.transpose(
                    np.reshape(
                        maskmem_enc,
                        (
                            maskmem_enc.shape[0],
                            maskmem_enc.shape[1],
                            maskmem_enc.shape[2] * maskmem_enc.shape[3],
                        ),
                    ),
                    (2, 0, 1),
                )
                # Temporal positional encoding
                maskmem_enc = (
                    maskmem_enc + self.maskmem_tpos_enc[self.num_maskmem - t_pos - 1]
                )
                to_cat_memory_pos_embed.append(maskmem_enc)

            # Construct the list of past object pointers
            if self.use_obj_ptrs_in_encoder:
                max_obj_ptrs_in_encoder = min(num_frames, self.max_obj_ptrs_in_encoder)
                # First add those object pointers from selected conditioning frames
                # (optionally, only include object pointers in the past during evaluation)
                if not self.training and self.only_obj_ptrs_in_the_past_for_eval:
                    ptr_cond_outputs = {
                        t: out
                        for t, out in selected_cond_outputs.items()
                        if (t >= frame_idx if track_in_reverse else t <= frame_idx)
                    }
                else:
                    ptr_cond_outputs = selected_cond_outputs
                pos_and_ptrs = [
                    # Temporal pos encoding contains how far away each pointer is from current frame
                    (abs(frame_idx - t), out["obj_ptr"])
                    for t, out in ptr_cond_outputs.items()
                ]
                # Add up to (max_obj_ptrs_in_encoder - 1) non-conditioning frames before current frame
                for t_diff in range(1, max_obj_ptrs_in_encoder):
                    t = frame_idx + t_diff if track_in_reverse else frame_idx - t_diff
                    if t < 0 or (num_frames is not None and t >= num_frames):
                        break
                    out = output_dict["non_cond_frame_outputs"].get(
                        t, unselected_cond_outputs.get(t, None)
                    )
                    if out is not None:
                        pos_and_ptrs.append((t_diff, out["obj_ptr"]))
                # If we have at least one object pointer, add them to the across attention
                if len(pos_and_ptrs) > 0:
                    pos_list, ptrs_list = zip(*pos_and_ptrs)
                    # stack object pointers along dim=0 into [ptr_seq_len, B, C] shape
                    obj_ptrs = np.stack(ptrs_list, axis=0)
                    # a temporal positional embedding based on how far each object pointer is from
                    # the current frame (sine embedding normalized by the max pointer num).
                    if self.add_tpos_enc_to_obj_ptrs:
                        t_diff_max = max_obj_ptrs_in_encoder - 1
                        tpos_dim = C if self.proj_tpos_enc_in_obj_ptrs else self.mem_dim
                        obj_pos = pos_list
                        obj_pos = get_1d_sine_pe(obj_pos / t_diff_max, dim=tpos_dim)
                        # obj_pos = self.obj_ptr_tpos_proj(obj_pos) # identity
                        obj_pos = obj_pos.unsqueeze(1).expand(-1, B, self.mem_dim)
                    else:
                        obj_pos = np.zeros(
                            (len(pos_list), B, self.mem_dim), dtype=np.float32
                        )
                    if self.mem_dim < C:
                        # split a pointer into (C // self.mem_dim) tokens for self.mem_dim < C
                        obj_ptrs = obj_ptrs.reshape(
                            -1, B, C // self.mem_dim, self.mem_dim
                        )
                        obj_ptrs = np.transpose(obj_ptrs, (0, 2, 1, 3))
                        obj_ptrs = np.reshape(
                            obj_ptrs,
                            (obj_ptrs.shape[0] * obj_ptrs.shape[1],)
                            + obj_ptrs.shape[2:],
                        )
                        obj_pos = np.repeat(obj_pos, C // self.mem_dim, axis=0)
                    to_cat_memory.append(obj_ptrs)
                    to_cat_memory_pos_embed.append(obj_pos)
                    num_obj_ptr_tokens = obj_ptrs.shape[0]
                else:
                    num_obj_ptr_tokens = 0
        else:
            # for initial conditioning frames, encode them without using any previous memory
            if self.directly_add_no_mem_embed:
                # directly add no-mem embedding (instead of using the transformer encoder)
                pix_feat_with_mem = current_vision_feats[-1] + self.no_mem_embed
                pix_feat_with_mem = np.transpose(pix_feat_with_mem, (1, 2, 0)).reshape(
                    B, C, H, W
                )
                return pix_feat_with_mem

            # Use a dummy token on the first frame (to avoid empty memory input to tranformer encoder)
            to_cat_memory = [self.no_mem_embed.expand(1, B, self.mem_dim)]
            to_cat_memory_pos_embed = [self.no_mem_pos_enc.expand(1, B, self.mem_dim)]

        # Step 2: Concatenate the memories and forward through the transformer encoder
        memory = np.concatenate(to_cat_memory, axis=0)
        memory_pos_embed = np.concatenate(to_cat_memory_pos_embed, axis=0)

        pix_feat = np.transpose(current_vision_feats[-1], (1, 2, 0)).reshape(
            (B, C, H, W)
        )
        memory_0 = memory[:-num_obj_ptr_tokens, :, :]
        memory_1 = memory[-num_obj_ptr_tokens:, :, :]
        memory_test = np.concatenate((memory_0, memory_1), axis=0)
        # 在此处将记忆进行帧进行填充，以到达可以静态输入的效果

        memory_0 = reflection_padding((28672, 1, 64), memory_0)
        memory_1 = reflection_padding((64, 1, 64), memory_1)

        memory_pos_embed = reflection_padding((28736, 1, 64), memory_pos_embed)

        pix_feat_with_mem = self.memory_attention.process(
            self.memory_attention_graph_name,
            {
                "current_vision_feat": pix_feat,
                "current_vision_pos_embed": current_vision_pos_embeds[0],
                "memory_pos_embed": memory_pos_embed,
                "memory_0": memory_0,
                "memory_1": memory_1,
            },
        )

        pix_feat_with_mem = pix_feat_with_mem["image_embed_Reshape_f32"]

        return pix_feat_with_mem

    def _prepare_backbone_features(self, backbone_out):
        """Prepare and flatten visual features."""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        assert len(backbone_out["backbone_fpn"]) >= self.num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-self.num_feature_levels :]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-self.num_feature_levels :]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC

        vision_feats = [
            np.transpose(
                np.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])),
                (2, 0, 1),
            )
            for x in feature_maps
        ]
        vision_pos_embeds = [
            np.transpose(
                np.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3])),
                (2, 0, 1),
            )
            for x in vision_pos_embeds
        ]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _apply_non_overlapping_constraints(self, pred_masks):
        """
        Apply non-overlapping constraints to the object scores in pred_masks. Here we
        keep only the highest scoring object at each spatial location in pred_masks.
        """
        batch_size = pred_masks.size(0)
        if batch_size == 1:
            return pred_masks

        # "max_obj_inds": object index of the object with the highest score at each location
        max_obj_inds = np.argmax(pred_masks, axis=0, keepdim=True)
        # "batch_obj_inds": object index of each object slice (along dim 0) in `pred_masks`
        batch_obj_inds = np.arange(batch_size)[:, None, None, None]
        keep = max_obj_inds == batch_obj_inds
        # suppress overlapping regions' scores below -10.0 so that the foreground regions
        # don't overlap (here sigmoid(-10.0)=4.5398e-05)
        pred_masks = np.where(keep, pred_masks, np.clip(pred_masks, max=-10.0))
        return pred_masks
