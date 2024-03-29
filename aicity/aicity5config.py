_base_ = ["../vid/temporal_roi_align/selsa_troialign_faster_rcnn_x101_dc5_7e_imagenetvid.py"]
 
dataset_type = "CocoVideoDataset" #'ImagenetVIDDataset'
data_root = 'data/aicity/'

model = dict(
    type='SELSA',
    detector=dict(
        roi_head=dict(
            type='SelsaRoIHead',
            bbox_roi_extractor=dict(
                type='TemporalRoIAlign',
                num_most_similar_points=2,
                num_temporal_attention_blocks=4,
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=2),
                out_channels=512,
                featmap_strides=[16]),
            bbox_head=dict(
                type='SelsaBBoxHead',
                num_shared_fcs=3,
                num_classes = 9,
                aggregator=dict(
                    type='SelsaAggregator',
                    in_channels=1024,
                    num_attention_blocks=16)))))


#classes 선언부 총 9개의 aicity dataset에 맞는 classes를 부여
CLASSES = ('motorbike', "DHelmet", 'DNoHelmet', 'P1Helmet', 'P1NoHelmet', 'P2Helmet', 'P2NoHelmet', 'P0Helmet', 'P0NoHelmet')

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqLoadAnnotations', with_bbox=True, with_track=True),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.5),
    dict(type='SeqBrightnessAug', jitter_range= 0.2),
    #dict(type='SeqBlurAug', prob= [0.0, 0.2]),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img', 'gt_bboxes', 'gt_labels', 'gt_instance_ids']),
    dict(type='ConcatVideoReferences'),
    dict(type='SeqDefaultFormatBundle', ref_prefix='ref')
]

test_pipeline = [
    dict(type='LoadMultiImagesFromFile'),
    dict(type='SeqResize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='SeqRandomFlip', share_params=True, flip_ratio=0.0),
    dict(type='SeqNormalize', **img_norm_cfg),
    dict(type='SeqPad', size_divisor=16),
    dict(
        type='VideoCollect',
        keys=['img'],
        meta_keys=('num_left_ref_imgs', 'frame_stride')),
    dict(type='ConcatVideoReferences'),
    dict(type='MultiImagesToTensor', ref_prefix='ref'),
    dict(type='ToList')
]

#data_loader 부분 여기를 본인의 경로에 맞게 설정하시면 됩니다.
data = dict(
    train=[
        dict(
            type = dataset_type,
            classes = CLASSES,
            ann_file=data_root + 'train/cocovid_train.json',
            img_prefix=data_root + 'train/images/',
            ref_img_sampler = dict(
                num_ref_imgs = 6,
                frame_range = [-8, 8],
                filter_key_img = True,
                method = 'bilateral_uniform'

            ),
            pipeline = train_pipeline 
            
            )
    ],
    val=dict(
        type = dataset_type,                
        classes = CLASSES,
        ann_file=data_root + 'train/cocovid_train.json',
        img_prefix=data_root + 'train/images/',
        pipeline = test_pipeline
        
        ),
    test=dict(
        type = dataset_type,
        classes = CLASSES,
        ann_file=data_root + 'train/cocovid_train.json',
        img_prefix=data_root + 'train/images/',
        pipeline = test_pipeline
        ))
