    

    

 

    # Example usage
    small_noise = 0.000316  # Small perturbation value
    large_noise = 0.02  # Larger perturbation for spikes
    spike_indices = [0, 1, 2, 3, 4]  # Indices where larger spikes occur
    p1, p2 = generate_spike(d, small_noise, large_noise, spike_indices)

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
         