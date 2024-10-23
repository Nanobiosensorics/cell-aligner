def show_alignment_result(ax, well_id, result):
  microscope_img, microscope_points = result[well_id][0]
  well_img, _, _ = result[well_id][1]
  translation = result[well_id][2]

  ax.set_axis_off()

  height1, width1 = microscope_img.shape[:2]
  height2, width2 = well_img.shape[:2]

  ax.imshow(microscope_img, cmap="gray")
  ax.scatter(
      microscope_points[:, 0], microscope_points[:, 1],
      color="red", alpha=0.25
  )
  ax.imshow(
      well_img, cmap="viridis", alpha=0.4,
      extent = [translation[0], translation[0]  + height2,
                translation[1] + width2, translation[1]]
  )

  ax.set_xlim([0, width1])
  ax.set_ylim([height1, 0])