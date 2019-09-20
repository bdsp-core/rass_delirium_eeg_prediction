# Code and examples for "Automated tracking of level of consciousness and delirium in critical illness using deep learning"

This repository contains the code, trained models, and examples of using the trained models, as used in the paper
[Sun, H., Kimchi, E., Akeju, O., Nagaraj, S.B., McClain, L.M., Zhou, D.W., Boyle, E., Zheng, W.L., Ge, W. and Westover, M.B., 2019. Automated tracking of level of consciousness and delirium in critical illness using deep learning. *npj Digital Medicine*, 2(1), pp.1-8](https://www.nature.com/articles/s41746-019-0167-0).

`Note 1`: We only tested these code in Python 2.7. Please use any conversion tool to convert to Python 3 if needed.
`Note 2`: The model is trained using PyTorch 0.4.0. There is incompatibility in the BatchNorm layer in the newer PyTorch version. A possible workaround is to change the keys in the model to the newer version.
`Note 3`: There are 3 files with >100MB which are not uploaded here. They should be downloaded from https://www.dropbox.com/sh/2t2n9ct4rik4l1p/AABXxhnDA0CWT1iO-r9j0wlsa?dl=0 and copy to paper_code/figures.
`Note 4`: The code in paper_code/figures are all runnable. But the other code is not runnable since we do not upload the whole dataset online.

## Contact

hsun 8 at mgh dot harvard dot edu

## License

This module is provided under the [MIT License](https://opensource.org/licenses/MIT).

