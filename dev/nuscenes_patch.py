import numpy as np


# Estimate the velocity for an annotation.
def box_velocity(nusc, curr_annotation_token: str, next_annotation_token: str) -> np.ndarray:

    curr = nusc.get('sample_annotation', curr_annotation_token)
    next = nusc.get('sample_annotation', next_annotation_token)

    pos_next = np.array(next['translation'])
    pos_curr = np.array(curr['translation'])
    pos_diff = pos_next - pos_curr

    time_next = 1e-6 * nusc.get('sample', next['sample_token'])['timestamp']
    time_curr = 1e-6 * nusc.get('sample', curr['sample_token'])['timestamp']
    time_diff = time_next - time_curr
    
    return pos_diff / time_diff


# Renders the box in the provided Matplotlib axis.
def render2d_box(axis,
            corners: np.ndarray,
            linecolor = 'k',
            linewidth = 2,
            linestyle = "solid",
            text = "",
            textcolor = 'k') -> None:

    # Draw the sides
    for i in range(3):
        axis.plot([corners[i][0], corners[i + 1][0]],
                    [corners[i][1], corners[i + 1][1]],
                    color=linecolor, linewidth=linewidth, linestyle=linestyle)

    axis.plot([corners[-1][0], corners[0][0]],
                [corners[-1][1], corners[0][1]],
                color=linecolor, linewidth=linewidth, linestyle=linestyle)

    if text:
        xpos = corners[0][0]
        ypos = corners[1][1] - 15
        axis.text(xpos, ypos, text, color=textcolor, fontsize="xx-large")