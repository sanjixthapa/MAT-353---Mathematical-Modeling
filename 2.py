import random

def estimate_pi(num_samples):
    inside_count = 0

    for _ in range(num_samples):
        x = random.uniform(-0.5, 0.5)
        y = random.uniform(-0.5, 0.5)

        if x ** 2 + y ** 2 <= 0.5 ** 2:
            inside_count = inside_count + 1

    pi_estimate = 4 * (inside_count / num_samples)
    return pi_estimate

num_samples = 1000000 #greater the sample, the better
approx_pi = estimate_pi(num_samples)
print(f"Approximation of π using {num_samples} samples: {approx_pi}")






