import decode
def simulateData(param, emitters, struct):

    frame_range_train = struct['framesize']

    psf = decode.utils.calibration_io.SMAPSplineCoefficient(
            calib_file=param.InOut.calibration_file).init_spline(
            xextent=struct['xextent'],
            yextent=struct['yextent'],
            img_shape=struct['img_shape'],
            device=param.Hardware.device_simulation,
            roi_size=param.Simulation.roi_size,
            roi_auto_center=param.Simulation.roi_auto_center
        )
    param.Simulation.bg_uniform = struct['bg']
    bg = decode.simulation.background.UniformBackground.parse(param)

    if param.CameraPreset == 'Perfect':
        noise = decode.simulation.camera.PerfectCamera.parse(param)
    elif param.CameraPreset is not None:
        raise NotImplementedError
    else:
        noise = decode.simulation.camera.Photon2Camera.parse(param)

    sim_data = decode.simulation.simulator.Simulation(psf=psf, background=bg,
                                                              noise=noise, frame_range=frame_range_train)
    frames, bg_frames, _ = sim_data.forward(emitters)
    return frames, bg_frames