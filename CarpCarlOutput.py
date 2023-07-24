import pandas as pd


def pass_to_fileout(df):
    new_df = df[['user_id', 'item_id', 'ratings']].rename(columns={
                                                                "user_id":"UserId", 
                                                                "item_id":"ItemId",
                                                                "ratings":"Rating"
                                                                                    })
    new_df['Date'] = [i for i in range(len(new_df))]
    return new_df


def get_document(df, col):
    new_df = df.groupby(by=col)['reviews'].sum().reset_index()
    
    new_df['reviews'] = new_df['reviews'].apply(lambda row: str(row).split())
    new_df['reviews'] = [" ".join(rev) for rev in new_df['reviews']]
    
    return new_df


if __name__ == "__main__":
    datasets_path = "./dataset/Video_Ga_data"
    
    data_path = f"{datasets_path}/data.csv"
    train_path = f"{datasets_path}/train/Train.csv"
    test_path = f"{datasets_path}/test/Test.csv"
    val_path = f"{datasets_path}/val/Val.csv"
    
    out_path = "./carl_carp_parl/"
    
    print("Charge datasets...")
    data = pd.read_csv(data_path)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    val = pd.read_csv(val_path)
    
    print("Pass train, test and val to fileout...")
    trainIterOut = pass_to_fileout(train)
    valIterOut = pass_to_fileout(val)
    testIterOut = pass_to_fileout(test)
    
    trainIterOut.to_csv(f"{out_path}/TrainInteraction.out", sep="\t", header=None, index=False)
    valIterOut.to_csv(f"{out_path}/ValInteraction.out", sep="\t", header=None, index=False)
    valIterOut.to_csv(f"{out_path}/ValidateInteraction.out", sep="\t", header=None, index=False)
    testIterOut.to_csv(f"{out_path}/TestInteraction.out", sep="\t", header=None, index=False)
    
    
    print("Creating the vocabulary...")
    vocab = {}

    i = 0
    for text in data['reviews']:
        text = str(text).split()
        for word in text:
            if word not in vocab.keys():
                vocab[word] = i
                i += 1
                
    vocab = pd.DataFrame({
        "words":list(vocab.keys()),
        "id": list(vocab.values())
        })
    
    vocab.to_csv(f"{out_path}/WordDict.out", sep="\t", header=None, index=False)
    
    
    print("Creating documents...")
    userReviewDocument = get_document(data, "user_id")
    itemReviewDocument = get_document(data, "item_id")
    
    userReviewDocument.to_csv(f"{out_path}/UserReviews.out", sep="\t", header=None, index=False)
    userReviewDocument.to_csv(f"{out_path}/UserAuxiliaryReviews.out", sep="\t", header=None, index=False)
    itemReviewDocument.to_csv(f"{out_path}/ItemReviews.out", sep="\t", header=None, index=False)