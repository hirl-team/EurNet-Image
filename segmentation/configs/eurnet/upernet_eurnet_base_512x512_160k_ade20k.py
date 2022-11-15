_base_ = [
    '../_base_/models/upernet_eurnet.py', '../_base_/datasets/ade20k.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]
model = dict(
    backbone=dict(
        type='EurNet',
        hidden_dim=[128, 256, 512, 1024], 
        depths=[2, 2, 18, 2],
        medium_stages=[1, 2, 3],
        edge_types=["short", "medium", "long"],
        num_neighbors=[12, 12, 12, 12],
        window_length=144,
        dilation=2,
        virtual_node=True,
        mlp_ratio=4.,
        drop_rate=0.,
        drop_path_rate=0.3,
        context_relation=True,
        context_sizes=[9,11,13],
        use_all_context=True,
        patch_norm=True,
        out_indices=(0, 1, 2, 3)),
    decode_head=dict(
        in_channels=[128, 256, 512, 1024],
        num_classes=150
    ),
    auxiliary_head=dict(
        in_channels=512,
        num_classes=150
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optimizer = dict(_delete_=True, type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)

# By default, models are trained on 8 GPUs with 2 images per GPU
data=dict(samples_per_gpu=2)
