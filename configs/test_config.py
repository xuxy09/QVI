testset_root = './datasets/example'
test_size = (854, 480)
test_crop_size = (854, 480)

mean = [0.429, 0.431, 0.397]
std  = [1, 1, 1]

inter_frames = 3


model = 'QVI'
pwc_path = './utils/pwc-checkpoint.pt'


store_path = 'outputs/example/'
checkpoint = 'qvi_release/model.pt'


