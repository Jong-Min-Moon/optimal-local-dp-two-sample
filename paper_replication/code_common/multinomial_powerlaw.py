    
    power=param_dist
    p1, p2 = create_power_law(d, power)

    server_private.load_private_data_multinomial(
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p1, sample_size),
            k,
            privacy_level,
            device
        ),
        LDPclient.release_private(
            priv_mech,
            data_gen.generate_multinomial_data(p2, sample_size),
            k,
            privacy_level,
            device
        ),
    d,
    device,
    device
    )
         