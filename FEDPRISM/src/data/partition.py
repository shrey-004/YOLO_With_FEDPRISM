import numpy as np
import torch

def partition_data(train_dataset, test_dataset, num_clients, partition_type, alpha=0.5):
    """
    Partitions the training and testing datasets for clients.
    Ensures the data distribution (class ratios) is consistent between train and test for each client.
    
    Args:
        train_dataset: The training dataset.
        test_dataset: The testing dataset.
        num_clients: Number of clients.
        partition_type: 'dirichlet' or 'pathological'.
        alpha: Alpha parameter for Dirichlet distribution.
        
    Returns:
        dict_users_train: Dictionary of train indices for each client.
        dict_users_test: Dictionary of test indices for each client.
    """
    # Get labels
    if hasattr(train_dataset, 'targets'):
        y_train = np.array(train_dataset.targets)
        y_test = np.array(test_dataset.targets)
    elif hasattr(train_dataset, 'labels'): # SVHN
        y_train = np.array(train_dataset.labels)
        y_test = np.array(test_dataset.labels)
    else:
        raise ValueError("Dataset does not have targets or labels attribute.")
        
    K = len(np.unique(y_train))
    N_train = y_train.shape[0]
    N_test = y_test.shape[0]
    
    dict_users_train = {i: [] for i in range(num_clients)}
    dict_users_test = {i: [] for i in range(num_clients)}

    if partition_type == 'dirichlet':
        # We iterate through classes and split them according to Dirichlet proportions
        # We use the SAME proportions for both Train and Test to ensure same distribution
        
        # To ensure we don't get empty clients, we might need to retry, 
        # but for simplicity and speed we'll implement the standard logic.
        
        min_size = 0
        min_require_size = 10
        
        # We need to find a valid distribution first (using Train as reference)
        while min_size < min_require_size:
            idx_batch_train = [[] for _ in range(num_clients)]
            idx_batch_test = [[] for _ in range(num_clients)]
            
            for k in range(K):
                idx_k_train = np.where(y_train == k)[0]
                idx_k_test = np.where(y_test == k)[0]
                
                np.random.shuffle(idx_k_train)
                np.random.shuffle(idx_k_test)
                
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                
                # Balance logic (from original code)
                proportions = np.array([p * (len(idx_j) < N_train / num_clients) for p, idx_j in zip(proportions, idx_batch_train)])
                proportions = proportions / proportions.sum()
                
                # Split Train
                proportions_train = (np.cumsum(proportions) * len(idx_k_train)).astype(int)[:-1]
                idx_batch_train = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_train, np.split(idx_k_train, proportions_train))]
                
                # Split Test (using same proportions)
                proportions_test = (np.cumsum(proportions) * len(idx_k_test)).astype(int)[:-1]
                idx_batch_test = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch_test, np.split(idx_k_test, proportions_test))]

            min_size = min([len(idx_j) for idx_j in idx_batch_train])
            
        for i in range(num_clients):
            dict_users_train[i] = idx_batch_train[i]
            dict_users_test[i] = idx_batch_test[i]

    elif partition_type == 'pathological':
        # 2 classes per client
        # We assign 2 random classes to each client.
        # Then we give them all samples of those classes (divided among clients sharing that class)
        
        # Original implementation used Shards. 
        # To map this to Test data, we need to know WHICH classes each client got.
        # The original implementation:
        # 1. Sorts data by label.
        # 2. Divides into 2*num_clients shards.
        # 3. Assigns 2 shards to each client.
        
        # This means each shard contains data from usually 1 or at most 2 classes.
        # To keep it simple and consistent for Test data:
        # Let's assign CLASSES directly.
        # Each client gets 2 distinct classes.
        # We distribute the data of those classes among the clients that selected them.
        
        # Determine classes for each client
        # We want to mimic the "Shard" behavior where classes are distributed evenly.
        # 200 shards for 100 clients. 10 classes. => 20 shards per class.
        # Each client gets 2 shards.
        
        # Let's stick to the Shard logic for Train, and infer classes for Test.
        
        # Train Shards
        num_shards = 2 * num_clients
        num_imgs_train = int(N_train / num_shards)
        idx_shard = [i for i in range(num_shards)]
        
        idxs_train = np.arange(num_shards * num_imgs_train)
        
        # Sort labels
        idxs_labels = np.vstack((idxs_train, y_train))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs_train_sorted = idxs_labels[0, :]
        
        # We also need to sort Test data by label to "shard" it similarly?
        # Or just assign test data based on the labels present in the train shards?
        # Let's do the latter. It's more robust.
        
        # 1. Assign Train Shards
        client_shards = {}
        for i in range(num_clients):
            rand_set = set(np.random.choice(idx_shard, 2, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            client_shards[i] = rand_set
            
            for rand in rand_set:
                dict_users_train[i] = np.concatenate((dict_users_train[i], idxs_train_sorted[rand*num_imgs_train:(rand+1)*num_imgs_train]), axis=0).astype(int)
        
        # 2. Assign Test Data
        # For each client, find the classes in their Train data
        for i in range(num_clients):
            train_indices = dict_users_train[i]
            client_labels = np.unique(y_train[train_indices])
            
            # Assign all Test data of these classes to this client?
            # No, multiple clients might have the same class.
            # We need to split the test data of class k among all clients that have class k.
            pass
            
        # Alternative Pathological for Test:
        # Just replicate the Shard logic on Test data.
        # Sort Test data. Divide into shards. Assign corresponding shards?
        # But "Corresponding" is hard to map.
        
        # Better approach:
        # 1. Create shards for Train. Assign to clients.
        # 2. Track which classes each shard corresponds to.
        # 3. Create shards for Test (same number). Assign to clients based on class match?
        
        # Simplest robust approach:
        # Use the "Proportions" logic but with extreme Dirichlet (alpha -> 0) or just manual assignment.
        # Let's rewrite Pathological to be explicit about classes.
        # "Pathological" usually means 2 classes per client.
        # We can just assign 2 classes to each client.
        # And split the data of those classes evenly among the clients that own them.
        
        # Let's do that. It's cleaner.
        
        # 1. Assign classes to clients
        # Ensure each class is assigned to roughly equal number of clients?
        # 100 clients, 2 classes each => 200 slots. 10 classes => 20 clients per class.
        
        classes = np.arange(K)
        client_classes = {i: [] for i in range(num_clients)}
        
        # Create a list of all available class slots
        slots = np.repeat(classes, 2 * num_clients // K) # Assuming 2*num_clients is divisible by K
        np.random.shuffle(slots)
        
        for i in range(num_clients):
            client_classes[i] = slots[i*2 : (i+1)*2]
            
        # 2. Distribute data
        # Group clients by class
        class_clients = {k: [] for k in range(K)}
        for i in range(num_clients):
            for c in client_classes[i]:
                class_clients[c].append(i)
                
        # Split data
        for k in range(K):
            idx_k_train = np.where(y_train == k)[0]
            idx_k_test = np.where(y_test == k)[0]
            
            np.random.shuffle(idx_k_train)
            np.random.shuffle(idx_k_test)
            
            clients_k = class_clients[k]
            num_k = len(clients_k)
            
            # Split evenly
            train_split = np.array_split(idx_k_train, num_k)
            test_split = np.array_split(idx_k_test, num_k)
            
            for j, client_idx in enumerate(clients_k):
                dict_users_train[client_idx].extend(train_split[j])
                dict_users_test[client_idx].extend(test_split[j])
                
    else:
        raise ValueError("Partition type not supported.")

    return dict_users_train, dict_users_test
