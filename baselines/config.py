mac_language_encoder_kwargs = {
    'encoder_type': 'lstm',
    'decoder_type': 'linear',
    'null_token': None,
    'encoder_vocab_size': None,
    'wordvec_dim': 300,
    'hidden_dim': 128,
    'rnn_num_layers': 1,
    'rnn_dropout': 0,
    'parameter_efficient': True,
    'output_batchnorm': False,
    'bidirectional': True,
    'gamma_option': 'linear', # not used
    'gamma_baseline': 1, # not used
    'use_attention': False, # not used
    'taking_context': True,
    'variational_embedding_dropout': 0.,
    'embedding_uniform_boundary': 1.0,
    'module_num_layers': 1,
    'num_modules': None, # from the mac model kwargs
}

film_language_encoder_kwargs = {
    'encoder_type': 'gru',
    'decoder_type': 'linear',
    'null_token': None,
    'encoder_vocab_size': None,
    'wordvec_dim': 200,
    'hidden_dim': 1024,
    'rnn_num_layers': 1,
    'rnn_dropout': 0,
    'parameter_efficient': True,
    'output_batchnorm': False,
    'bidirectional': False,
    'gamma_option': 'linear', # not used
    'gamma_baseline': 1, # not used
    'use_attention': False, # not used
    "num_modules": None,
}

mac_kwargs = {
    'vocab': None,
    'feature_dim': None, # raw images: [3,128,128]; resnet18 features: [256, 14, 14]
    'stem_num_layers': 6,
    'stem_batchnorm': True,
    'stem_kernel_size': [3],
    'stem_subsample_layers': [1,3], # add MaxPool2d(kernel_size=2, stride=2)
    'stem_stride': [1],
    'stem_padding': None,
    'stem_dim': 64,
    'num_modules': None,
    'module_dim': 128,
    'question_embedding_dropout': 0.,
    'stem_dropout': 0.,
    'memory_dropout': 0.,
    'read_dropout': 0.,
    'nonlinearity': 'ELU',
    'use_prior_control_in_control_unit': 0 == 1,
    'use_self_attention': 0,
    'use_memory_gate': 0,
    'question2output': 1,
    'classifier_batchnorm': 0 == 1,
    'classifier_fc_layers': [1024],
    'classifier_dropout': 0.,
    'use_coords': 1, # 1, 0
    'write_unit': 'original',
    'read_connect': 'last',
    'noisy_controls': bool(0),
    'debug_every': float('inf'),
    'print_verbose_every': float('inf'),
    'hard_code_control' : False
    }
    
film_kwargs = {
    'vocab': None,
    'feature_dim': None, # raw images: [3,128,128]; resnet18 features: [256, 14, 14]
    'stem_num_layers': 6,
    'stem_batchnorm': True,
    'stem_kernel_size': [3],
    'stem_subsample_layers': [1,3], # add MaxPool2d(kernel_size=2, stride=2)
    'stem_stride': [1],
    'stem_padding': None,
    'stem_dim': 64,
    'num_modules': None,
    'module_num_layers': 1,
    'module_dim': 64, #was 128 in original FiLM`
    'module_residual': True,
    'module_intermediate_batchnorm': False,
    'module_batchnorm': True,
    'module_batchnorm_affine': False,
    'module_dropout': 0e-2,
    'module_input_proj': 1,
    'module_kernel_size': 3,
    'classifier_proj_dim': 512,
    'classifier_downsample': 'maxpool2',
    'classifier_fc_layers': [1024],
    'classifier_batchnorm': True,
    'classifier_dropout': 0,
    'condition_method': 'bn-film',
    'condition_pattern': None, # [1,1,1,1], # length should be equal to "num_modules"
    'use_gamma': True,
    'use_beta': True,
    'use_coords': 1, # 1, 0
    'debug_every': float('inf'),
    'print_verbose_every': float('inf'),
}

output_kwargs = {
        "record_loss_every_n_iter": 10,
        'evaluation_every_n_iter': 1000,
        "checkpoint_every_n_iter": 50000,
        'root_datadir': None,
    }

train_kwargs = {
    "total_epochs": None,
    "num_val_samples": 10 ** 4,
    # "total_iterations": 300000,
}