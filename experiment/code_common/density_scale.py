    
    copula_mean = torch.zeros(d).to(device)

    copula_sigma_1 = (0.5 * torch.ones(d,d) + 0.5 * torch.eye(d)).to(device)
    copula_sigma_2 = copula_sigma_1.mul(5)
    
    data_y_priv = LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean, copula_sigma_1),
            privacy_level,
            n_bin,
            device
        )

    data_z_priv = LDPclient.release_private_conti(
            priv_mech,
            data_gen.generate_copula_gaussian_data(sample_size, copula_mean, copula_sigma_2),
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