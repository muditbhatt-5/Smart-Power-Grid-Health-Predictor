import pandas as pd
import numpy as np

def extract_data_from_image(image_path):
    # Simulate extraction logic - replace this with actual ML model for table extraction
    # For now, generating dummy data similar to what would be extracted
    # Replace this section with integration to any ML table extraction model (like Donut, TableNet, etc.)
    data = {
        'Voltage': np.random.randint(100, 400, 10),
        'Current': np.random.randint(10, 50, 10),
        'Temperature': np.random.randint(25, 80, 10),
        'Load': np.random.randint(200, 800, 10)
    }
    df = pd.DataFrame(data)
    return df