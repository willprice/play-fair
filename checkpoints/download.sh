#!/usr/bin/env bash
set -euo pipefail
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"


download() {
  dir="$1"; shift
  url="$1"; shift
  wget --content-disposition -P "$dir" "$url"
}

echo "-----------------------"
echo "Downloading full models"
echo "-----------------------"
echo "Downloading BN-Inception Multiscale TRN trained on SSv2"
# Thanks Bolei and Alex for the great models!
# this is repacked to be consistent with our data
# source: https://github.com/zhoubolei/TRN-pytorch/
download "$DIR/backbones" "https://www.dropbox.com/s/u9ajcv13ndljo8i/trn.pth?dl=1"

echo "Downloading ResNet-50 TSN trained on SSv2"
download "$DIR/backbones" "https://www.dropbox.com/s/nnf826io52k5kq4/tsn.pth?dl=1"

echo "--------------------------"
echo "Downloading feature models"
echo "--------------------------"
download "$DIR/features" "https://www.dropbox.com/s/mzjiz32haf3dfnj/mtrn_8_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/915nr87r031jmmv/mtrn_16_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/dcxij3ccc402hrt/trn_1_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/wsslrmg5r7n8cbv/trn_2_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/86bfh3undhjdc6y/trn_3_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/01ypt4bvsqz7t3q/trn_4_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/nhvzu5qo1d8ws2j/trn_5_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/g4omdta1cbbvlbh/trn_6_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/ie8ifymetpsynma/trn_7_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/5xxkjtrk4m0ogy0/trn_8_frames.pth?dl= 1"
download "$DIR/features" "https://www.dropbox.com/s/64ia4gud5mk3evq/trn_9_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/4fwdsd1zpbv6sem/trn_10_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/06rcq0e1jx8mw9j/trn_11_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/zph1aevb1e2i7v8/trn_12_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/d8bb5b0ceqokiq0/trn_13_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/h1zrtzv53suhji5/trn_14_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/1fot7tunzteju75/trn_15_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/v7baj7aue9oa65a/trn_16_frames.pth?dl=1"
download "$DIR/features" "https://www.dropbox.com/s/ziivu7dto228n4e/tsn.pth?dl=1"
