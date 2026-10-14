# Model Configurations

## SR (Spectral Residual)
- Library: OpenCV 4.x
- Function: cv2.saliency.StaticSaliencySpectralResidual_create()
- Parameters: default (n=3, Gaussian 8x8, sigma=2.5)
- Downsampling: max dimension 64 pixels

## Itti-Koch
- Library: pysaliency 0.2.22
- Implementation: IttiKoch
- Parameters: default (9 scales, 4 orientations at 0°, 45°, 90°, 135°)

## GBVS
- Library: gbvs (shreelock/gbvs, GitHub)
- Parameters: salmapmaxsize=32, blurfrac=0.02, channels='DIO', gaborangles=[0,45,90,135]
