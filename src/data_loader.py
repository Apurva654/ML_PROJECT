import os
import pandas as pd
import numpy as np

MUMBAI_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "mumbai_house_prices.csv")

# Actual benchmark price bands across MMR (Average rate per sq. ft. in INR)
MUMBAI_MARKET_RATES = {
    'South Mumbai (Colaba / Marine Drive / Malabar Hill)': 68000,
    'Worli / Lower Parel / Prabhadevi': 46000,
    'Bandra West / Khar / Santacruz West': 48000,
    'Juhu / Vile Parle West': 42000,
    'Andheri West / Lokhandwala': 26000,
    'Powai / Hiranandani': 23500,
    'Andheri East / Marol / Sakinaka': 17500,
    'Goregaon / Malad West': 19500,
    'Borivali / Kandivali West': 17000,
    'Ghatkopar / Mulund West': 18500,
    'Chembur / Wadala': 21000,
    'Thane West (Ghodbunder Road)': 11000,
    'Thane (Majiwada / Teen Hath Naka)': 14500,
    'Navi Mumbai (Vashi / Palm Beach Road)': 18500,
    'Navi Mumbai (Kharghar / Belapur / Nerul)': 11500,
    'Mira Road / Bhayandar': 8500,
    'Kalyan / Dombivli / Beyond Thane': 5800
}

def generate_realistic_mumbai_data(n_samples=2500, seed=42):
    """
    Generates realistic market listings calibrated to actual Mumbai Metropolitan Region prices.
    Prices are output in INR Lakhs (e.g., 150.0 = ₹1.5 Crore).
    """
    np.random.seed(seed)
    
    localities = list(MUMBAI_MARKET_RATES.keys())
    weights = [0.03, 0.04, 0.05, 0.04, 0.09, 0.08, 0.08, 0.09, 0.09, 0.07, 0.06, 0.10, 0.06, 0.05, 0.05, 0.04, 0.04]
    weights = np.array(weights) / sum(weights)
    
    locality = np.random.choice(localities, size=n_samples, p=weights)
    bhk = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.28, 0.44, 0.20, 0.06, 0.02])
    
    carpet_area = []
    bathrooms = []
    balcony = []
    
    for b in bhk:
        if b == 1:
            area = np.random.randint(340, 520)
            bath = 1
            balc = np.random.choice([0, 1], p=[0.4, 0.6])
        elif b == 2:
            area = np.random.randint(620, 950)
            bath = 2
            balc = np.random.choice([1, 2], p=[0.3, 0.7])
        elif b == 3:
            area = np.random.randint(1050, 1550)
            bath = np.random.choice([2, 3], p=[0.15, 0.85])
            balc = np.random.choice([1, 2, 3], p=[0.2, 0.5, 0.3])
        elif b == 4:
            area = np.random.randint(1700, 2700)
            bath = 4
            balc = np.random.randint(2, 4)
        else: # 5 BHK luxury
            area = np.random.randint(3000, 5200)
            bath = 5
            balc = np.random.randint(3, 5)
            
        carpet_area.append(area)
        bathrooms.append(bath)
        balcony.append(balc)
        
    carpet_area = np.array(carpet_area)
    floor_no = np.random.randint(1, 42, size=n_samples)
    total_floors = floor_no + np.random.randint(2, 18, size=n_samples)
    property_age = np.random.choice([0, 1, 3, 6, 10, 15, 25], size=n_samples, p=[0.20, 0.20, 0.25, 0.15, 0.10, 0.06, 0.04])
    
    # Mumbai Amenities
    gym = np.random.choice([1, 0], size=n_samples, p=[0.65, 0.35])
    swimming_pool = np.random.choice([1, 0], size=n_samples, p=[0.42, 0.58])
    clubhouse = np.random.choice([1, 0], size=n_samples, p=[0.55, 0.45])
    car_parking = np.random.choice([1, 0], size=n_samples, p=[0.75, 0.25])
    gated_security = np.random.choice([1, 0], size=n_samples, p=[0.90, 0.10])
    metro_distance = np.round(np.random.uniform(0.3, 9.0, size=n_samples), 1)
    
    prices_lakhs = []
    for i in range(n_samples):
        loc = locality[i]
        base_rate = MUMBAI_MARKET_RATES[loc]
        
        # Amenities premium
        amenity_mult = 1.0 + (0.04 * gym[i]) + (0.05 * swimming_pool[i]) + (0.03 * clubhouse[i]) + (0.04 * car_parking[i])
        
        # Floor rise (higher floors command slight view/light premium in Mumbai highrises)
        floor_mult = 1.0 + (0.002 * min(floor_no[i], 30))
        
        # Age discount
        age_mult = max(0.80, 1.0 - (0.008 * property_age[i]))
        
        # Metro accessibility
        metro_mult = 1.04 if metro_distance[i] <= 1.2 else (0.97 if metro_distance[i] > 5.0 else 1.0)
        
        # Market variance noise (±5%)
        noise = np.random.normal(1.0, 0.05)
        
        total_inr = (carpet_area[i] * base_rate) * amenity_mult * floor_mult * age_mult * metro_mult * noise
        # 1 Lakh = 100,000 INR
        lakhs = round(total_inr / 100000.0, 2)
        prices_lakhs.append(lakhs)
        
    df = pd.DataFrame({
        'Locality': locality,
        'BHK': bhk,
        'CarpetArea_sqft': carpet_area,
        'Bathrooms': bathrooms,
        'Balcony': balcony,
        'FloorNo': floor_no,
        'TotalFloors': total_floors,
        'PropertyAge_Years': property_age,
        'MetroDistance_KM': metro_distance,
        'Gymnasium': gym,
        'SwimmingPool': swimming_pool,
        'Clubhouse': clubhouse,
        'CarParking': car_parking,
        'GatedSecurity': gated_security,
        'Price_In_Lakhs': prices_lakhs
    })
    return df

def load_housing_data(custom_path=None):
    if custom_path:
        df = pd.read_csv(custom_path)
        for target in ['Price_In_Lakhs', 'price', 'SalePrice']:
            if target in df.columns:
                return df, "Custom Uploaded Dataset", target
        raise ValueError("Uploaded CSV must have a recognized price column.")
        
    # Always guarantee the Mumbai dataset is generated and saved locally
    if not os.path.exists(MUMBAI_DATA_PATH):
        os.makedirs(os.path.dirname(MUMBAI_DATA_PATH), exist_ok=True)
        df = generate_realistic_mumbai_data()
        df.to_csv(MUMBAI_DATA_PATH, index=False)
    else:
        df = pd.read_csv(MUMBAI_DATA_PATH)
        
    return df, "Mumbai Metropolitan Region (MMR) Real Estate Dataset", "Price_In_Lakhs"