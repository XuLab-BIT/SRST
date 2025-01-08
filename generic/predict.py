import decode
def work(param, frames, model):
    mirror_frame = True
    camera = decode.simulation.camera.Photon2Camera.parse(param)
    camera.device = 'cpu'
    device = 'cuda:0'

    post_raw_th = param.PostProcessingParam.raw_th
    frame_proc = decode.neuralfitter.utils.processing.TransformSequence([
        # decode.neuralfitter.utils.processing.wrap_callable(camera.backward),
        decode.neuralfitter.frame_processing.AutoCenterCrop(8),
        decode.neuralfitter.scale_transform.AmplitudeRescale.parse(param)
    ])

    size_procced = decode.neuralfitter.frame_processing.get_frame_extent(frames.unsqueeze(1).size(), frame_proc.forward)  # frame size after processing
    frame_extent = ((0, size_procced[-2]), (0, size_procced[-1]))

    post_proc = decode.neuralfitter.utils.processing.TransformSequence([
        
        decode.neuralfitter.scale_transform.InverseParamListRescale.parse(param),
        
        decode.neuralfitter.coord_transform.Offset2Coordinate(xextent=frame_extent[0],
                                                            yextent=frame_extent[1],
                                                            img_shape=size_procced[-2:]),
        
        decode.neuralfitter.post_processing.SpatialIntegration(raw_th=post_raw_th,
                                                            xy_unit='px', 
                                                            px_size=param.Camera.px_size)
    ])
    infer_ts = decode.neuralfitter.Infer(model=model, ch_in=param.HyperParameter.channels_in,
                                    frame_proc=frame_proc, post_proc=post_proc,
                                    device=device, num_workers=0, pin_memory=False,
                                    batch_size='auto')
    em_pred = infer_ts.forward(frames)

    return em_pred