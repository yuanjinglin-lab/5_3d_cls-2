import torch
import torch.nn.functional as F

def torch_SSIM(image_true, image_test, data_range=255., window_size=11, sigma=1.5):
    # Calculate constants for SSIM
    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # Create a Gaussian window
    window = torch.exp(-(torch.linspace(-(window_size // 2), window_size // 2, window_size) ** 2) / (2.0 * sigma ** 2))
    window = window / window.sum()

    # Compute local means
    mu_true = F.conv2d(image_true, window.view(1, 1, window_size, 1), padding=0, groups=1)
    mu_test = F.conv2d(image_test, window.view(1, 1, window_size, 1), padding=0, groups=1)

    # Compute local variances
    mu_true_sq = mu_true * mu_true
    mu_test_sq = mu_test * mu_test
    mu_true_test = mu_true * mu_test

    # Compute local variances
    sigma_true_sq = F.conv2d(image_true * image_true, window.view(1, 1, window_size, 1), padding=0, groups=1) - mu_true_sq
    sigma_test_sq = F.conv2d(image_test * image_test, window.view(1, 1, window_size, 1), padding=0, groups=1) - mu_test_sq
    sigma_true_test = F.conv2d(image_true * image_test, window.view(1, 1, window_size, 1), padding=0, groups=1) - mu_true_test

    # Compute SSIM map
    ssim_map = ((2 * mu_true_test + C1) * (2 * sigma_true_test + C2)) / ((mu_true_sq + mu_test_sq + C1) * (sigma_true_sq + sigma_test_sq + C2))

    # Return the mean SSIM value of the entire image
    return ssim_map.mean()

# Example usage:
image_true = torch.rand(1, 3, 256, 256)  # Replace with your true image
image_test = torch.rand(1, 3, 256, 256)  # Replace with your test image
ssim = torch_SSIM(image_true, image_test)
print("SSIM:", ssim.item())




