#!/usr/bin/env python3
import numpy as np
import os

# =================================================================
# CONFIGURATION
# =================================================================
# Update 'data_dir' to the folder containing your .txt files
data_dir = './Project/data_in_matrixForm/' 
file_test  = os.path.join(data_dir, 'testTrack_hierarchy.txt')
file_train = os.path.join(data_dir, 'trainIdx2_matrix.txt')
file_out   = os.path.join(data_dir, 'output1.txt')

def process_music_data():
    # Verify files exist before starting
    if not os.path.exists(file_test) or not os.path.exists(file_train):
        print(f"Error: Ensure {file_test} and {file_train} exist.")
        return

    print("Starting data processing...")

    # Open files using 'with' blocks to ensure they close automatically
    with open(file_test, 'r') as fTest, \
         open(file_train, 'r') as fTrain, \
         open(file_out, 'w') as fOut:

        # Initialize the first line of the training file
        train_line = fTrain.readline()
        
        # Buffers to hold data for the current user's 6 test tracks
        track_ids = [0] * 6
        album_ids = [0] * 6
        artist_ids = [0] * 6
        current_user_id = None
        item_index = 0
        
        # Matrix to store ratings: 6 tracks x 2 types (Album Rating, Artist Rating)
        # We use zeros as a default "no rating found" value
        user_ratings = np.zeros(shape=(6, 2))

        for line in fTest:
            # Parse Test File: userID | trackID | albumID | artistID
            test_parts = line.strip().split('|')
            if len(test_parts) < 4: continue
            
            u_id, t_id, al_id, ar_id = test_parts
            
            # Reset buffers if we encounter a new User ID
            if u_id != current_user_id:
                item_index = 0
                user_ratings.fill(0)
                current_user_id = u_id
            
            # Store hierarchy info for the current track
            track_ids[item_index] = t_id
            album_ids[item_index] = al_id
            artist_ids[item_index] = ar_id
            item_index += 1
            
            # Once we've collected exactly 6 tracks for this user, search the training file
            if item_index == 6:
                while train_line:
                    train_parts = train_line.strip().split('|')
                    if len(train_parts) < 3: 
                        train_line = fTrain.readline()
                        continue
                        
                    train_u_id, train_item_id, train_rating = train_parts
                    
                    # Optimization: Skip training data that precedes our current test user
                    if train_u_id < current_user_id:
                        train_line = fTrain.readline()
                        continue
                    
                    # If IDs match, check if this item is one of the user's 6 albums or artists
                    if train_u_id == current_user_id:
                        for n in range(6):
                            if train_item_id == album_ids[n]:
                                user_ratings[n, 0] = train_rating # Store Album Rating
                            if train_item_id == artist_ids[n]:
                                user_ratings[n, 1] = train_rating # Store Artist Rating
                        train_line = fTrain.readline()
                    
                    # If we've passed the user in the training file, write results and move to next test user
                    else:
                        for n in range(6):
                            output = f"{current_user_id}|{track_ids[n]}|{user_ratings[n,0]}|{user_ratings[n,1]}\n"
                            fOut.write(output)
                        break 

    print(f"Processing complete. Results saved to: {file_out}")

if __name__ == "__main__":
    process_music_data()
    