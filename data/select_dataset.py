def define_Dataset(dataset_opt):
    dataset_type = dataset_opt['dataset_type'].lower()
    if dataset_type in ['acdc_2d']:
        # from data.datasets.dataset_ACDC_2D import Dataset_ACDC_2D as D
        from data.datasets.ACDC.dataset_ACDC_2D_with_properties import Dataset_ACDC_2D as D
    elif dataset_type in ['acdc_2d_square']:
        from data.datasets.ACDC.dataset_ACDC_2D_square import Dataset_ACDC_2D as D
    elif dataset_type in ['acdc_2d_with_classification']:
        from data.datasets.ACDC.dataset_ACDC_2D_with_properties_classification import Dataset_ACDC_2D as D
    elif dataset_type in ['brats_2d']:
        from data.datasets.BraTS.dataset_BraTS_2D_square import Dataset_BraTS_2D as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not found.'.format(dataset_type))

    dataset = D(dataset_opt)
    print('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__, dataset_opt['name']))
    return dataset
