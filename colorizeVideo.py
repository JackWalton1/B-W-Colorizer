"""The purpose of this file is to try to abstact away the file paths
from colorizeVideo.py and turn a b&w video into color"""
import argparse
import matplotlib.pyplot as plt

from colorizers import *
from utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')

# Add path arguments
parser.add_argument('-vn','--video_name', type=str, default='GrapesOfWrathTrailerClip',
                    help='Name of the video clip')

opt = parser.parse_args()

input_frames_folder = "./frames/"+ opt.video_name+"/output/"
eccv_frames_folder = "./frames/"+opt.video_name+"/eccv/"
siggraph_frames_folder = "./frames/"+opt.video_name+"/siggraph/"
eccv_output_video = "./videos_out/eccv/"+opt.video_name+".mp4"
siggraph_output_video = "./videos_out/siggraph/"+opt.video_name+".mp4"

video_path = "./videos/" +opt.video_name+".mp4"
output_folder = "./frames/" +opt.video_name + "/output"

#b&w video -> b&w frames
video_to_frames(video_path, output_folder)

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# Example usage
i = 0
# b&w frames to colorized frames
for i in range(countFrames(input_frames_folder)):
    code = int_to_str_with_leading_zeros(i)
    defaultPath = input_frames_folder+ 'frame_' + str(code) + '.png'

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

    # Save colorized frames using eccv_output_folder and siggraph_output_folder
    if not os.path.exists(eccv_frames_folder):
        print("No folder named: ", eccv_frames_folder,"\nMaking the folder.")
        os.makedirs(eccv_frames_folder)
    if not os.path.exists(siggraph_frames_folder):
        print("No folder named: ", siggraph_frames_folder,"\nMaking the folder.")
        os.makedirs(siggraph_frames_folder)
    plt.imsave(os.path.join(eccv_frames_folder, 'colorFrame_%s.png' % code), out_img_eccv16)
    plt.imsave(os.path.join(siggraph_frames_folder, 'colorFrame_%s.png' % code), out_img_siggraph17)


#colorized franes to colorized video
frames_to_video(eccv_frames_folder, eccv_output_video)
frames_to_video(siggraph_frames_folder, siggraph_output_video)