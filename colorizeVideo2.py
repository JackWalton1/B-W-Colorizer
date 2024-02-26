"""The purpose of this file is to try to abstact away the file paths
from colorizeVideo.py and turn a b&w video into color"""
import argparse
import matplotlib.pyplot as plt

from colorizers import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')

# Add path arguments
parser.add_argument('-iff','--input_frames_folder', type=str, default='./frames/GrapesOfWrathTrailerClip/output/',
                    help='Path to the folder containing input frames')
parser.add_argument('-eof','--eccv_output_folder', type=str, default='./frames/GrapesOfWrathTrailerClip/eccv/',
                    help='Path to the folder for ECCV16 colorized frames')
parser.add_argument('-sof','--siggraph_output_folder', type=str, default='./frames/GrapesOfWrathTrailerClip/siggraph/',
                    help='Path to the folder for SIGGRAPH17 colorized frames')
parser.add_argument('-eov','--eccv_output_video', type=str, default='./videos_out/eccv/GrapesOfWrathTrailerClipColor.mp4',
                    help='Path to the output video for ECCV16 colorization')
parser.add_argument('-sov','--siggraph_output_video', type=str, default='./videos_out/siggraph/GrapesOfWrathTrailerClipColor.mp4',
                    help='Path to the output video for SIGGRAPH17 colorization')

opt = parser.parse_args()
# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# Example usage
i = 0
# for i in range(countFrames('./frames/GrapesOfWrathTrailerClip/output/')):
for i in range(countFrames(opt.input_frames_folder)):
    code = int_to_str_with_leading_zeros(i)
    defaultPath = opt.input_frames_folder+ 'frame_' + str(code) + '.png'

    # default size to process images is 256x256
    # grab L channel in both original ("orig") and resized ("rs") resolutions
    img = load_img(defaultPath)
    (tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
    if(opt.use_gpu):
        tens_l_rs = tens_l_rs.cuda()
        
    # colorizer outputs 256x256 ab map
    # resize and concatenate to original L channel
    img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
    out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
    out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

    # Save colorized frames using opt.eccv_output_folder and opt.siggraph_output_folder
    plt.imsave(os.path.join(opt.eccv_output_folder, 'colorFrame_%s.png' % code), out_img_eccv16)
    plt.imsave(os.path.join(opt.siggraph_output_folder, 'colorFrame_%s.png' % code), out_img_siggraph17)


eccv_input_folder = "./frames/GrapesOfWrathTrailerClip/eccv"
siggraph_input_folder = "./frames/GrapesOfWrathTrailerClip/siggraph"
eccv_output_video_path = "./videos_out/eccv/GrapesOfWrathTrailerClipColor.mp4"
siggraph_output_video_path = "./videos_out/siggraph/GrapesOfWrathTrailerClipColor.mp4"

# print(input_folder)
# print(output_video_path)

frames_to_video(eccv_input_folder, eccv_output_video_path)
frames_to_video(siggraph_input_folder, siggraph_output_video_path)