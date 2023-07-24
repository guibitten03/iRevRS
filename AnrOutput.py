import pandas as pd


if __name__ == "__main__":
    
    def charge_data(path, name):
        print("Charge datasets...")
        data = pd.read_csv(path)
        data['date'] = [i for i in range(len(data))]
        
        out = f"./ANR/datasets/{name}"
        
        data.to_json(out, orient="records", lines=True)
        
    
    dataset = "./dataset/Video_Ga_data"
    
    data_path = f"{dataset}/data.csv"
    
    name = "yelp_video_game_data.json"
    
    charge_data(data_path, name)