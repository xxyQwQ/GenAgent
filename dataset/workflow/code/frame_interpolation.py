# create nodes by instantiation
vhs_videocombine_3 = VHS_VideoCombine(frame_rate=24, loop_count=0, filename_prefix="""AnimateDiff""", format="""image/gif""", pingpong=False, save_output=True)
vhs_loadvideo_7 = VHS_LoadVideo(video="""play_guitar.gif""", force_rate=0, force_size="""Disabled""", custom_width=512, custom_height=512, frame_load_cap=0, skip_first_frames=0, select_every_nth=1)
rife_vfi_10 = RIFE_VFI(ckpt_name="""rife47.pth""", clear_cache_after_n_frames=10, multiplier=3, fast_mode=True, ensemble=True, scale_factor=1)

# link nodes by invocation
image_7, frame_count_7, audio_7, video_info_7 = vhs_loadvideo_7(meta_batch=None)
image_10 = rife_vfi_10(frames=image_7, optional_interpolation_states=None)
filenames_3 = vhs_videocombine_3(images=image_10, audio=None, meta_batch=None)
