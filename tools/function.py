
import torch
import tqdm
def train(model, train_loader, loss_function, optimizer, device, cfg):
    model.train()
    for i, (data, label) in enumerate(tqdm(train_loader, desc="Train batches")):
            data = data.to(device)
            label = label.to(device)
            out = model(data)

            out = torch.sigmoid(out)
            loss = loss_function(out, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return 0 


def validate(model, val_loader, device, cfg):
    model.eval()
    num_correct = 0
    num_pixels = 0
    with torch.no_grad():
         for i, (data, label) in enumerate(tqdm(val_loader, desc="Validate batches")):
            data = data.to(device)
            label = label.to(device)
            out_img = model(data)

            probability = torch.sigmoid(out_img)
            predictions = probability > 0.5
            num_correct += (predictions==label).sum()
            num_pixels += cfg['TEST']['BATCH_SIZE_PER_GPU'] * cfg['DATASET']['IMG_WIDTH'] * cfg['DATASET']['IMG_HEIGHT']

    return num_correct / num_pixels
            
        
              


