
    copula_mean_1 = -0.5 * torch.ones(d).to(device)
    copula_mean_2 =  -copula_mean_1


    copula_sigma = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(device)
    data_y_priv = LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_1, copula_sigma),
            privacy_level,
            n_bin,
            device
        )

    data_z_priv = LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean_2, copula_sigma),
            privacy_level,
            n_bin,
            device
        )
    server_private.load_private_data_multinomial(
        data_y_priv,
        data_z_priv,
        LDPclient.alphabet_size_binned,
        device,
        device
    )