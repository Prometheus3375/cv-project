param([String]$dir)
(Get-ChildItem "$dir/*/" -Directory) | Foreach-Object { python test_segmentation_deeplab.py -i $_ }
