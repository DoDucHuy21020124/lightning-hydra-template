import pyrootutils
import gradio as gr

import sys
import pyrootutils
root = pyrootutils.setup_root(__file__, pythonpath=True)
sys.path.append(root)

from workspace.src.filter_functions import *

# examples
example_1_dir = str(root/ "workspace/inputs/images/filters/eye_glasses.png")
example_2_dir = str(root / "workspace/inputs/images/filters/le_hai_lam.png")
example_3_dir = str(root / "workspace/inputs/images/filters/leonardo_dicarpio.jpg")
example_list = [[example_1_dir],[example_2_dir],[example_3_dir]]

# for app
filter_names = ["eye_glasses", "face_swapping"]

# Create the Gradio demo
image_tab = gr.Interface(
    fn= filter_image,
    inputs=[gr.Image(), gr.inputs.Radio(choices=filter_names, label="Select a filter:")],
    outputs=gr.Image(type="pil"),
    examples=example_list
)

video_tab = gr.Interface(
    fn= filter_video,
    inputs=[gr.Video(source = 'upload'), gr.inputs.Radio(choices=filter_names, label="Select a filter:")],
    outputs=gr.Video(type="pil")
)

demo = gr.TabbedInterface([image_tab, video_tab], ["Image", "Video"])

# Launch the demo!
demo.launch(debug=False, # print errors locally?
            share=False) # generate a publically shareable URL?

# launch the demo with docker
# demo.launch()