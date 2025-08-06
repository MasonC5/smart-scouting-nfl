import cfbd
from cfbd.rest import ApiException
from cfbd.configuration import Configuration
from cfbd.api.stats_api import StatsApi
import pandas as pd
import os

# Configure your API key securely
configuration = Configuration(access_token="UuYSK0L32c+HuwG1djun5KBsp7qOFX5m5I5xzsmuL77MYJAWvIXpz5KyVKR2LTst")

all_stats = []

with cfbd.ApiClient(configuration) as api_client:
    stats_api = StatsApi(api_client)
    for year in range(2010, 2026):  # 2010 through 2025
        try:
            response = stats_api.get_player_season_stats(year=year)
            all_stats.extend([s.to_dict() for s in response])
            print(f"Fetched {len(response)} records for {year}")
        except ApiException as e:
            print(f"Error fetching {year}: {e}")

# Convert fetched data to DataFrame
df = pd.DataFrame(all_stats)

# Ensure output directory exists
os.makedirs("data", exist_ok=True)
output_path = "data/cfb_player_stats_2010_2025.csv"
df.to_csv(output_path, index=False)

print(f"Saved college player stats (2010–2025) to {output_path}")
