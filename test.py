import SimpleITK as sitk
import numpy as np
import torch
import numpy as np

def produceRandomlyDeformedImage(sitkImage, sitklabel, numcontrolpoints, stdDef):
    transfromDomainMeshSize=[numcontrolpoints]*sitkImage.GetDimension()

    tx = sitk.BSplineTransformInitializer(sitkImage, transfromDomainMeshSize)

    params = tx.GetParameters()

    paramsNp=np.asarray(params,dtype=float)
    paramsNp = paramsNp + np.random.randn(paramsNp.shape[0])*stdDef

    paramsNp[0:int(len(params)/3)]=0 #remove z deformations! The resolution in z is too bad

    params=tuple(paramsNp)
    tx.SetParameters(params)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitkImage)
    resampler.SetDefaultPixelValue(0)
    resampler.SetTransform(tx)

    resampler.SetInterpolator(sitk.sitkLinear)
    outimgsitk = resampler.Execute(sitkImage)

    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    outlabsitk = resampler.Execute(sitklabel)

    # save
    sitk.WriteImage(outimgsitk, r'D:/image-test.nii')
    sitk.WriteImage(outlabsitk, r'D:/label-test.nii')


if __name__ == "__main__":
    # image = sitk.ReadImage(r'D:\Projects\OrgansSegment\Data\Sample\image\image0000.nii.gz')
    # label = sitk.ReadImage(r'D:\Projects\OrgansSegment\Data\Sample\label\label0000.nii')
    # print(image.GetDimension())
    # produceRandomlyDeformedImage(image, label, 4, 5)

    t = torch.tensor(np.array([[0.9, 0.5], [1.0, 0.8]]), dtype=torch.float16).cuda()
    t1 = torch.tensor(np.ones(t.shape)*0.9, dtype=torch.float16).cuda()
    # print(t1.numpy())
    t2 = 0.9 - torch.relu(0.9-t)
    print(t2.cpu().numpy())