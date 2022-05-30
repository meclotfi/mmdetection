_base_=[
'../_base_/default_runtime.py']
model = dict(
    type='DeformableDETR',
    backbone=dict(
        type='MobileViT',
        out_indices=(2,3,4),
        Layers_config=dict(
            layer1=dict(
                type='mobilenet2',
                out_channels=24,
                expand_ratio=4,
                num_blocks=1,
                stride=1),
            layer2=dict(
                type='mobilenet2',
                out_channels=48,
                expand_ratio=1,
                num_blocks=3,
                stride=2),
            layer3=dict(
                type='mobilevit',
                out_channels=64,
                head_dim=32,
                ffn_dim=64,
                n_transformer_blocks=2,
                patch_h=8,
                patch_w=8,
                stride=2,
                mv_expand_ratio=2,
                num_heads=2,
                expand_ratio=4),
            layer4=dict(
                type='mobilevit',
                out_channels=80,
                head_dim=16,
                ffn_dim=64,
                n_transformer_blocks=4,
                patch_h=2,
                patch_w=2,
                stride=2,
                mv_expand_ratio=2,
                num_heads=4,
                expand_ratio=1),
            layer5=dict(
                type='mobilevit',
                out_channels=96,
                head_dim=16,
                ffn_dim=128,
                n_transformer_blocks=4,
                patch_h=8,
                patch_w=8,
                stride=2,
                mv_expand_ratio=2,
                num_heads=4,
                expand_ratio=1)),
        init_cfg=dict(type='Pretrained',
            checkpoint='torchvision://resnet50')),
    neck=dict(
        type='ChannelMapper',
        in_channels=[64,80,96],
        kernel_size=1,
        out_channels=64,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=4),
    bbox_head=dict(
        type='DeformableDETRHead',
        num_query=50,
        num_classes=1,
        in_channels=128,
        sync_cls_avg_factor=True,
        as_two_stage=False,
        transformer=dict(
            type='DeformableDetrTransformer',
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=64),
                    ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=64,
                     feedforward_channels=128,
                     num_fcs=2,
                 ),
                    feedforward_channels=128,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
            decoder=dict(
                type='DeformableDetrTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=64,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=64)
                    ],
                    ffn_cfgs=dict(
                     type='FFN',
                     embed_dims=64,
                     feedforward_channels=128,
                     num_fcs=2,
                 ),
                    feedforward_channels=128,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=32,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0)),
        # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='HungarianAssigner',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBoxL1Cost', weight=5.0, box_format='xywh'),
            iou_cost=dict(type='IoUCost', iou_mode='giou', weight=2.0))),
    test_cfg=dict(max_per_img=50))

# dataset settings
dataset_type = 'VwwDetection'
data_root = './'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=dict(backend='http')),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Expand',
        mean=img_norm_cfg['mean'],
        to_rgb=img_norm_cfg['to_rgb'],
        ratio_range=(1, 2)),
    dict(
        type='MinIoURandomCrop',
        min_ious=(0.4, 0.5, 0.6, 0.7, 0.8, 0.9),
        min_crop_size=0.3),
    dict(
        type='Resize',
        img_scale=(320, 320),
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile',file_client_args=dict(backend='http')),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(320, 320),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=8,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_person_ids="annotations/person.npy",
        img_prefix="coco/annotations/instances_train2017.json",
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix='coco/annotations/instances_val2017.json',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox',metric_options={'topk': (1,)})

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    weight_decay=0.0001,
    )
optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(policy='step', step=[40])
runner = dict(type='EpochBasedRunner', max_epochs=50)