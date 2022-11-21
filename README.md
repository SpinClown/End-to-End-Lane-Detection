# End-to-End-Lane-Detection
The evaluation code of a polynomial based end-to-end lane detection method.

The evaluation code is modified from [Ultra-Fast-Lane-Detection](https://github.com/cfzd/Ultra-Fast-Lane-Detection)

# Install
Please see INSTALL.md

# Evaluation
We provide the Res-34-3rd model on TuSimple.

[BaiduDrive(code:mb0n)](https://pan.baidu.com/s/1cBGbkIswB_ehRjMvQBaceA) 

```Shell
python test.py configs/tusimple.py --test_model path_to_model.pth --test_work_dir ./tmp
```
