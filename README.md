# 2023-Eval-POCUS-ONSD

This repository contains the code and notebooks used in: Moore BT, Osika T, Satterly S, Shah S, Thirion T, Hampton S, Aylward S, Montgomery S. Evaluation of commercially available point-of-care ultrasound for automated optic nerve sheath measurement. Ultrasound J. 2023 Aug 2;15(1):33. doi: 10.1186/s13089-023-00331-8. PMID: 37530991; PMCID: PMC10397168. [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10397168/](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10397168/)

Data is available upon request ([brad.moore@kitware.com](mailto:brad.moore@kitware.com)).

The package [/python/usqc/](/python/usqc/) contains code for making image quality measurements from images of the
CIRS 040GSE ultrasound phantom [https://www.cirsinc.com/products/ultrasound/multi-purpose-multi-tisse-ultrasound-phantom/](https://www.cirsinc.com/products/ultrasound/multi-purpose-multi-tisse-ultrasound-phantom/)

The current algorithm relies on segmentations (for the paper, manual segmentations) of the phantom objects in the image.  We implemented
the method this way as we noticed device-specific biases in the performance of the original image registration-based method.

Basic Approach:
1. Ultrasound images were preprocessed with itkpocus [https://pypi.org/project/itk-pocus/](https://pypi.org/project/itk-pocus/) (cropped and spatial dimensions set, converted to .mha format).
2. Used 3D Slicer [https://www.slicer.org/](https://www.slicer.org/) to manually segment phantom objects (wires, contrast targets) using the documentation in [/python/usqc/phantom.py](/python/usqc/phantom.py)
3. Used `PhantomRegistration` [/python/usqc/registration.py](/python/usqc/registration.py) to align segmentations to coordinates on the ultrasound phantom
4. The notebooks [/python/Analyze Contrast Fisher.ipynb](/python/Analyze%20Contrast%20Fisher.ipynb), [/python/Analyze PointSpread.ipynb](/python/Analyze%20PointSpread.ipynb), and [/python/Analyze SNR.ipynb](/python/Analyze%20SNR.ipynb) show how to calculate image metrics

This work was developed by Kitware [https://www.kitware.com](https://www.kitware.com).  Kitware provides R&D and software development services.  If you are interesting in collaborating on research related to this work or
procuring Kitware's services, please reach out [https://www.kitware.com/contact/](https://www.kitware.com/contact/).

This effort was sponsored by the U.S. Government under Other Transactions Number W81XWH-15-9-0001/W81XWH-19-9-0015 and the Medical Technology Enterprise Consortium (MTEC) under 19-08-MuLTI-0079. The views, opinions and/or findings contained in this publication are those of the authors and do not necessarily reflect the views of the Department of Defense (DoD) and should not be construed as an official DoD position, policy or decision unless so designated by other documentation. No official endorsement should be made.