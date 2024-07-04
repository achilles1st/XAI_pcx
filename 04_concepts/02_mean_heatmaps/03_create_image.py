import os
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

this_file_path = os.path.dirname(os.path.realpath(__file__))

for class_name_1, class_name_2, range_name in [
  ('Both', None, '__difference_top_bottom'),
  ('Both', None, '__top'),
  ('Both', None, '__bottom'),
  # ('AD', None),
  # ('NC', None),
  # ('AD', 'NC'),
  # ('NC', 'AD'),
]:
  target_class_name_2 = class_name_2 or 'None'
  target_base_path = os.path.join(this_file_path, f'{class_name_1}_diff_{target_class_name_2}_concepts_mean__{range_name}')
  if not os.path.exists(target_base_path):
    os.makedirs(target_base_path)

  data_base_path = f'/mnt/neuro/nas2/Work/Graz_plus_R2star/04_concepts/01_heatmaps/deep_taylor_decomposition/relevance/ref_images/R2star_RG/{class_name_1}_warped_mean_10'

  if class_name_2 is not None:
    data_base_path = f'/mnt/neuro/nas2/Work/Graz_plus_R2star/04_concepts/01_heatmaps/deep_taylor_decomposition/relevance/ref_images/R2star_RG/{class_name_1}_diff_{class_name_2}_warped_mean'
    if not os.path.exists(data_base_path):
      os.makedirs(data_base_path)

    data_base_path_1 = f'/mnt/neuro/nas2/Work/Graz_plus_R2star/04_concepts/01_heatmaps/deep_taylor_decomposition/relevance/ref_images/R2star_RG/{class_name_1}_warped_mean'
    data_base_path_2 = f'/mnt/neuro/nas2/Work/Graz_plus_R2star/04_concepts/01_heatmaps/deep_taylor_decomposition/relevance/ref_images/R2star_RG/{class_name_2}_warped_mean'

    mean_concepts_1 = os.listdir(data_base_path_1)
    mean_concepts_1.sort(reverse=True)

    mean_concepts_2 = os.listdir(data_base_path_2)
    mean_concepts_2.sort(reverse=True)

    for mean_concept_index, mean_concept_1 in enumerate(mean_concepts_1):
      params = [
       'fslmaths',
       os.path.join(data_base_path_1, mean_concepts_1[mean_concept_index]),
       '-sub',
       os.path.join(data_base_path_2, mean_concepts_2[mean_concept_index]),
       os.path.join(data_base_path, mean_concepts_1[mean_concept_index]),
      ]
      print(' '.join(params))
      subprocess.call(params)

  mean_concepts = os.listdir(data_base_path)
  mean_concepts.sort(reverse=True)
  mean_concepts.insert(0, '/mnt/neuro/nas2/Work/Graz_plus_R2star/03_results/02_performance/heatmaps_warped/R2star_RG/boostrap_index-05__initial_weights_index-02/sum.nii.gz')

  concepts_ordered = []

  count_conecpts = 0

  for mean_concept in mean_concepts:
    if count_conecpts > 0 and not mean_concept.endswith(f'{range_name}.nii.gz'):
      continue

    count_conecpts += 1
    if count_conecpts > 5:
      break

    if count_conecpts > 1:
      mean_concept_path = os.path.join(data_base_path, mean_concept)
      sprite_name = mean_concept[0:len(mean_concept)-7]
    else:
      mean_concept_path = mean_concept
      sprite_name = 'dtd_full'

    # mins
    whole_image = nib.load(mean_concept_path)
    whole_image_data = whole_image.get_fdata().flatten()

    whole_image_data_logical = whole_image_data > 0.
    whole_image_data = whole_image_data[whole_image_data_logical]

    whole_image_sum = np.sum(whole_image_data)
    whole_image_data_normalized = whole_image_data / whole_image_sum

    (n, bins, _) = plt.hist(whole_image_data_normalized,
                            bins=1000,
                            density=False,
                          )

    mean_bins = [(i + j) * 0.5 for i, j in zip(bins[:-1], bins[1:])]

    relevance_sum = .0
    min_i_list = []
    
    scales = [.50]
    scales_index = 0
    
    for bin_index, relevance_in_bin in reversed(list(enumerate(n * mean_bins))):
      relevance_sum = relevance_sum + relevance_in_bin
      if (relevance_sum >= scales[scales_index]):
        min_i_list.append(mean_bins[bin_index] * whole_image_sum)
        scales_index = scales_index + 1
        if scales_index == len(scales):
          break

    # max
    fsl_stats_args = [
      'fslstats',
      mean_concept_path,
      '-R',
    ]

    # print(' '.join(fsl_stats_args))
    p = subprocess.Popen(fsl_stats_args, stdout=subprocess.PIPE)
    output = p.stdout.readline()
    # print(output.decode())

    max_i = float(output.decode().strip().split(' ')[1])
    print(str(max_i))

    if max_i <= 0:
      max_i = 0
      min_i_list = [0.]

    convert_append_outer = [
      'convert',
    ]

    for min_i in min_i_list:
      convert_append_inner = [
        'convert',
      ]
      
      for slice_index in [62, 71, 76, 82, 94]: # [27, 73]:
        output_filepath = os.path.join(target_base_path, sprite_name + '_slice_' + str(slice_index) + '_from_histo.png')
        cropped_output_filepath = os.path.join(target_base_path, sprite_name + '_slice_' + str(slice_index) + '_from_histo_cropped.png')
        convert_append_inner.append(cropped_output_filepath)

        if min_i >= 0 and max_i > 0:
          fsleyes_rendering_args = [
            'fsleyes',
            'render',
            '--layout',
            'horizontal',
            # '--showColourBar',
            # '--colourBarLocation',
            # 'left',
            '--labelSize',
            '0',
            '--performance',
            '3',
            '--hideCursor',
            '--hideLabels',
            '--hidex',
            '--hidey',
            '--voxelLoc',
            '90',
            '108',
            str(slice_index),
            '--xcentre',
            '0',
            '0',
            '--ycentre',
            '0',
            '0',
            '--zcentre',
            '0',
            '0',
            '--outfile',
            output_filepath,
            # '--size',
            # '1920',
            # '1080',
            os.path.join('/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz'),
            # '--cmap',
            # 'brain_colours_1hot',
            mean_concept_path,
            '--cmap',
            'hot',
            '--displayRange',
            str(min_i),
            str(max_i),
            # outlines
            '/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz',
            '--overlayType',
            'label',
            '--lut',
            'graz_plus_basal_ganglia',
            '--outline',
            '--outlineWidth',
            '7',
          ]
        else:
          fsleyes_rendering_args = [
            'fsleyes',
            'render',
            '--layout',
            'horizontal',
            # '--showColourBar',
            # '--colourBarLocation',
            # 'left',
            '--labelSize',
            '0',
            '--performance',
            '3',
            '--hideCursor',
            '--hideLabels',
            '--hidex',
            '--hidey',
            '--voxelLoc',
            '90',
            '108',
            str(slice_index),
            '--xcentre',
            '0',
            '0',
            '--ycentre',
            '0',
            '0',
            '--zcentre',
            '0',
            '0',
            '--outfile',
            output_filepath,
            # '--size',
            # '1920',
            # '1080',
            os.path.join('/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz'),
            # '--cmap',
            # 'brain_colours_1hot',
            # mean_concept_path,
            # '--cmap',
            # 'hot',
            # '--displayRange',
            # str(min_i),
            # str(max_i),
            # outlines
            # '/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz',
            # '--overlayType',
            # 'label',
            # '--lut',
            # 'graz_plus_subcortical',
            # '--outline',
            # '--outlineWidth',
            # '6',
          ]

        # print(' '.join(fsleyes_rendering_args))
        subprocess.call(fsleyes_rendering_args)

        crop_args = [
          'convert',
          output_filepath,
          '-crop',
          '490x586+152+6',
          cropped_output_filepath
        ]
        # print(' '.join(crop_args))
        subprocess.call(crop_args)
      
      convert_append_inner.append('+append')
      convert_append_inner_output = os.path.join(target_base_path, sprite_name + '_sprite_from_histo.png')
      convert_append_inner.append(convert_append_inner_output)

      # print(' '.join(convert_append_inner))
      subprocess.call(convert_append_inner)

      convert_append_outer.append(convert_append_inner_output)
    
    convert_append_outer.append('-append')
    convert_append_outer_output = os.path.join(target_base_path, sprite_name + '.png')
    convert_append_outer.append(convert_append_outer_output)

    print(' '.join(convert_append_outer))
    subprocess.call(convert_append_outer)

    concepts_ordered.append(convert_append_outer_output)

    rel_index = len('rel_')
    channel_number_index = len('rel_0.351__channel_1')
    create_label_args = [
      'convert',
      '-size',
      '586x46',
      'canvas:black',
      '-pointsize',
      str(46),
      '-fill',
      'white',
      '-draw',
      f'gravity Center font DejaVu-Sans text 0,0 "concept {count_conecpts-1}, {float(mean_concept[rel_index:rel_index+5]):.0%}"'\
      if count_conecpts > 1 else f'gravity Center font DejaVu-Sans text 0,0 "z\u207A-rule"',
      '-rotate',
      '-90',
    ]
    create_label_args.append(os.path.join(target_base_path, sprite_name + '__channel_index.png'))
    print(' '.join(create_label_args))
    subprocess.call(create_label_args)

    c1_args = [
      'convert',
      os.path.join(target_base_path, sprite_name + '__channel_index.png'),
      convert_append_outer_output,
      '-background',
      'white',
      '-splice',
      '10x0+0+0',
      '+append',
      '-chop',
      '10x0+0+0',
      convert_append_outer_output,
      # os.path.join(target_base_path, 'row_1.png'),
    ]
    subprocess.call(c1_args)

  c_f_args = [
    'convert',
    # os.path.join(central_figure_path, 'header_1.png'),
    # concepts_ordered[0],
    concepts_ordered[1],
    concepts_ordered[2],
    concepts_ordered[3],
    concepts_ordered[4],
    # concepts_ordered[5],
    # concepts_ordered[6],
    # concepts_ordered[7],
    '-background',
    'white',
    '-splice',
    '0x10+0+0',
    '-append',
    '-chop',
    '0x10+0+0',
    os.path.join(target_base_path, 'central_image.png'),
  ]
  subprocess.call(c_f_args)

  c_f_args = [
    'convert',
    concepts_ordered[0],
    os.path.join(target_base_path, 'central_image.png'),
    '-background',
    'white',
    '-splice',
    '0x20+0+0',
    '-append',
    '-chop',
    '0x20+0+0',
    os.path.join(target_base_path, 'central_image.png'),
  ]
  subprocess.call(c_f_args)

    # add_text_args = [
    #   'convert',
    #   convert_append_outer_output,
    #   # '-gravity',
    #   # 'North',
    #   '-pointsize',
    #   '60',
    #   '-fill',
    #   'yellow',
    #   '-draw',
    #   'text 540,70 "' + class_acc_text + '"',
    #   convert_append_outer_output,
    # ]
    # print(' '.join(add_text_args))
    # subprocess.call(add_text_args)

  # # atlas
  # convert_append_inner = [
  #   'convert',
  # ]
      
  # for slice_index in [62, 71, 76, 82, 94]: # [27, 73]:
  #   output_filepath = os.path.join(target_base_path, 'atlas_sprite_slice_' + str(slice_index) + '.png')
  #   cropped_output_filepath = os.path.join(target_base_path, 'atlas_sprite_slice_' + str(slice_index) + '_cropped.png')
  #   convert_append_inner.append(cropped_output_filepath)

  #   fsleyes_rendering_args = [
  #     'fsleyes',
  #     'render',
  #     '--layout',
  #     'horizontal',
  #     # '--showColourBar',
  #     # '--colourBarLocation',
  #     # 'left',
  #     '--labelSize',
  #     '0',
  #     '--performance',
  #     '3',
  #     '--hideCursor',
  #     '--hideLabels',
  #     '--hidex',
  #     '--hidey',
  #     '--voxelLoc',
  #     '90',
  #     '108',
  #     str(slice_index),
  #     '--xcentre',
  #     '0',
  #     '0',
  #     '--ycentre',
  #     '0',
  #     '0',
  #     '--zcentre',
  #     '0',
  #     '0',
  #     '--outfile',
  #     output_filepath,
  #     # '--size',
  #     # '1920',
  #     # '1080',
  #     os.path.join('/opt/fsl/data/standard/MNI152_T1_1mm.nii.gz'),
  #     # '--cmap',
  #     # 'brain_colours_1hot',
  #   ]

  #   for (atlas, cmap, display) in [
  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'green', 4), # Left Thalamus
  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'green', 15), # Right Thalamus

  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'blue', 5), # Left Caudate nucleus
  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'blue', 16), # Right Caudate nucleus

  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'yellow', 6), # Left Putamen
  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'yellow', 17), # Right Putamen

  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'pink', 7), # Left Pallidum
  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'pink', 18), # Right Pallidum

  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'copper', 9), # Left Hippocampus
  #     ('/opt/fsl/data/atlases/HarvardOxford/HarvardOxford-sub-maxprob-thr25-1mm.nii.gz', 'copper', 19), # Right Hippocampus
  #   ]:
  #     fsleyes_rendering_args += [    
  #       atlas,
  #       '--cmap',
  #       cmap,
  #       '--linkHighRanges',
  #       '--displayRange',
  #       str(display - 1e-4),
  #       str(display + 1e-4),
  #     ]

  #   # print(' '.join(fsleyes_rendering_args))
  #   subprocess.call(fsleyes_rendering_args)

  #   crop_args = [
  #     'convert',
  #     output_filepath,
  #     '-crop',
  #     '490x586+152+6',
  #     cropped_output_filepath
  #   ]
  #   # print(' '.join(crop_args))
  #   subprocess.call(crop_args)
      
  # convert_append_inner.append('+append')
  # convert_append_inner_output = os.path.join(target_base_path, 'atlas_sprites.png')
  # convert_append_inner.append(convert_append_inner_output)

  # # print(' '.join(convert_append_inner))
  # subprocess.call(convert_append_inner)

  # for text, file_name, size, rotate in [
  #   # ('unmasked', 'unmasked.png', '3920x320', False),
  #   # ('masked', 'masked.png', '3920x320', False),
  #   # ('Graz\u207A', 'graz_plus.png', '3920x320', False),
  #   ('native', 'native.png', '586x46', True),
  #   ('linear', 'linear.png', '586x46', True),
  #   ('nonlinear', 'nonlinear.png', '586x46', True),
  #   ('atlas', 'atlas.png', '586x46', True),
  # ]:
  #   create_label_args = [
  #     'convert',
  #     '-size',
  #     size,
  #     'canvas:black',
  #     '-pointsize',
  #     str(46),
  #     '-fill',
  #     'white',
  #     '-draw',
  #     f'gravity Center font DejaVu-Sans text 0,0 "{text}"',
  #   ]
  #   if rotate:
  #     create_label_args.append('-rotate')
  #     create_label_args.append('-90')
  #   create_label_args.append(os.path.join(target_base_path, file_name))
  #   print(' '.join(create_label_args))
  #   subprocess.call(create_label_args)

  # # create_filer_args = [
  # #     'convert',
  # #     '-size',
  # #     '320x320',
  # #     'canvas:white',
  # #     os.path.join(central_figure_path, 'filler.png'),
  # # ]
  # # print(' '.join(create_filer_args))
  # # subprocess.call(create_filer_args)

  # # h1_args = [
  # #   'convert',
  # #   os.path.join(central_figure_path, 'filler.png'),
  # #   os.path.join(central_figure_path, 'unmasked.png'),
  # #   os.path.join(central_figure_path, 'masked.png'),
  # #   os.path.join(central_figure_path, 'graz_plus.png'),
  # #   '-background',
  # #   'white',
  # #   '-splice',
  # #   '40x0+0+0',
  # #   '+append',
  # #   '-chop',
  # #   '40x0+0+0',
  # #   os.path.join(central_figure_path, 'header_1.png'),
  # # ]
  # # subprocess.call(h1_args)

  # c1_args = [
  #   'convert',
  #   os.path.join(target_base_path, 'native.png'),
  #   # os.path.join(central_figure_path, 'R2star.png'),
  #   # os.path.join(central_figure_path, 'T1_BET.png'),
  #   os.path.join(target_base_path, 'R2star_RG.png'),
  #   '-background',
  #   'white',
  #   '-splice',
  #   '10x0+0+0',
  #   '+append',
  #   '-chop',
  #   '10x0+0+0',
  #   os.path.join(target_base_path, 'row_1.png'),
  # ]
  # subprocess.call(c1_args)

  # c2_args = [
  #   'convert',
  #   os.path.join(target_base_path, 'linear.png'),
  #   # os.path.join(central_figure_path, 'T1@MNI152_dof6.png'),
  #   # os.path.join(central_figure_path, 'T1@MNI152_dof6_BET.png'),
  #   os.path.join(target_base_path, 'R2star@MNI152_dof6_RG.png'),
  #   '-background',
  #   'white',
  #   '-splice',
  #   '10x0+0+0',
  #   '+append',
  #   '-chop',
  #   '10x0+0+0',
  #   os.path.join(target_base_path, 'row_2.png'),
  # ]
  # subprocess.call(c2_args)

  # c3_args = [
  #   'convert',
  #   os.path.join(target_base_path, 'nonlinear.png'),
  #   # os.path.join(central_figure_path, 'T1@MNI152_nlin_with_T1_mask.png'),
  #   # os.path.join(central_figure_path, 'T1@MNI152_nlin_BET_with_T1_mask.png'),
  #   os.path.join(target_base_path, 'R2star@MNI152_nlin_RG.png'),
  #   '-background',
  #   'white',
  #   '-splice',
  #   '10x0+0+0',
  #   '+append',
  #   '-chop',
  #   '10x0+0+0',
  #   os.path.join(target_base_path, 'row_3.png'),
  # ]
  # subprocess.call(c3_args)

  # c4_args = [
  #   'convert',
  #   os.path.join(target_base_path, 'atlas.png'),
  #   os.path.join(target_base_path, 'atlas_sprites.png'),
  #   '-background',
  #   'white',
  #   '-splice',
  #   '10x0+0+0',
  #   '+append',
  #   '-chop',
  #   '10x0+0+0',
  #   os.path.join(target_base_path, 'row_4.png'),
  # ]
  # subprocess.call(c4_args)

  # c_f_args = [
  #   'convert',
  #   # os.path.join(central_figure_path, 'header_1.png'),
  #   os.path.join(target_base_path, 'row_1.png'),
  #   os.path.join(target_base_path, 'row_2.png'),
  #   os.path.join(target_base_path, 'row_3.png'),
  #   os.path.join(target_base_path, 'row_4.png'),
  #   '-background',
  #   'white',
  #   '-splice',
  #   '0x10+0+0',
  #   '-append',
  #   '-chop',
  #   '0x10+0+0',
  #   os.path.join(target_base_path, 'central_image.png'),
  # ]
  # subprocess.call(c_f_args)
