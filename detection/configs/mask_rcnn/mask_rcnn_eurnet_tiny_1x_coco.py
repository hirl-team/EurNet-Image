_base_ = [
    '../_base_/models/mask_rcnn_eurnet_fpn.py',
    '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]

model = dict(
    backbone=dict(
        type='EurNet',
        hidden_dim=[96, 192, 384, 768],
        depths=[2, 2, 6, 2], 
        medium_stages=[1, 2, 3],
        edge_types=["short", "medium", "long"],
        num_neighbors=[12, 12, 12, 12],
        window_length=144,
        dilation=2,
        virtual_node=True,
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.1,
        context_relation=True,
        context_sizes=[9,11,13],
        use_all_context=True,
        patch_norm=True,
        out_indices=(0, 1, 2, 3)), 
    neck=dict(in_channels=[96, 192, 384, 768])) 

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='AutoAugment',
         policies=[
             [
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                                 (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                                 (736, 1333), (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True)
             ],
             [
                 dict(type='Resize',
                      img_scale=[(400, 1333), (500, 1333), (600, 1333)],
                      multiscale_mode='value',
                      keep_ratio=True),
                 dict(type='RandomCrop',
                      crop_type='absolute_range',
                      crop_size=(384, 600),
                      allow_negative_crop=True),
                 dict(type='Resize',
                      img_scale=[(480, 1333), (512, 1333), (544, 1333),
                                 (576, 1333), (608, 1333), (640, 1333),
                                 (672, 1333), (704, 1333), (736, 1333),
                                 (768, 1333), (800, 1333)],
                      multiscale_mode='value',
                      override=True,
                      keep_ratio=True)
             ]
         ]),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]

data_root = 'datasets/coco/'
data = dict(
    train=dict(pipeline=train_pipeline), 
    test=dict(
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/'
    ),    
    samples_per_gpu=1,
)

optimizer = dict(_delete_=True, type='AdamW', lr=0.00008, betas=(0.9, 0.999), weight_decay=0.05,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))
lr_config = dict(step=[9, 11])
runner = dict(type='EpochBasedRunnerAmp', max_epochs=12)

fp16 = None

optimizer_config = dict(
    type="DistOptimizerHook",
    update_interval=1,
    grad_clip=None,
    coalesce=True,
    bucket_size_mb=-1,
    use_fp16=True,
)