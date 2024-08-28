def check_processed_data(processed_subjets, batch_index=0):
    print(f"\n--- Checking Processed Data for Batch Item {batch_index} ---")
    print(f"Processed shape: {processed_subjets.shape}")
    if len(processed_subjets.shape) == 3:
        num_subjets, num_features, subjet_length = processed_subjets.shape
        print(f"Number of subjets: {num_subjets}")
        print(f"Number of features: {num_features}")
        print(f"Subjet length: {subjet_length}")
        print("\nFirst few values of each feature:")
        for i in range(num_features):
            print(f"Feature {i}: {processed_subjets[0, i, :5]}")
    else:
        print("Unexpected shape for processed subjets")

def create_random_masks(batch_size, num_subjets, num_features, subjet_length, context_scale=0.7):
    print(f"Creating random masks with batch_size={batch_size}, num_subjets={num_subjets}")
    context_masks = []
    target_masks = []

    for i in range(batch_size):
        indices = torch.randperm(num_subjets)
        context_size = int(num_subjets * context_scale)
        context_indices = indices[:context_size]
        target_indices = indices[context_size:]

        context_mask = torch.zeros(num_subjets, num_features, subjet_length)
        target_mask = torch.zeros(num_subjets, num_features, subjet_length)

        context_mask[context_indices] = 1
        target_mask[target_indices] = 1

        context_masks.append(context_mask)
        target_masks.append(target_mask)

    return torch.stack(context_masks), torch.stack(target_masks)

def train_step(model, particles, subjets, subjet_masks, particle_masks, optimizer, device, step):
    print(f"\nStarting training step {step}")
    
    # Debug Statement
    # check_processed_data(sjs)

    # represent subjets in terms of particles
    particle_indices = 
    subjet_particles = particles[:, :, particle_indices[0, subjet_idx].long()]
    
    batch_size, num_subjets, num_features, subjet_length = subjets.size()
    print(f"Input shapes - Subjets: {subjets.shape}, Subjet masks: {subjet_masks.shape}, Particle masks: {particle_masks.shape}")
    
    context_masks, target_masks = create_random_masks(batch_size, num_subjets, num_features, subjet_length)
    print(f"Context masks shape: {context_masks.shape}, Target masks shape: {target_masks.shape}")
    
    context_masks = context_masks.to(device)
    target_masks = target_masks.to(device)
    subjet_masks = subjet_masks.to(device)
    particle_masks = particle_masks.to(device)
    
    context_subjets = subjets * context_masks
    target_subjets = subjets * target_masks
    
    optimizer.zero_grad()
    
    print("Forwarding through model")
    pred_repr, context_repr, target_repr = model(context_subjets, target_subjets)
    
    print(f"Predicted representation shape: {pred_repr.shape}")
    print(f"Target representation shape: {target_repr.shape}")
    
    combined_mask = target_masks.to(device) * subjet_masks.unsqueeze(-1).unsqueeze(-1).expand_as(target_masks).to(device)
    
    pred_repr = pred_repr.to(device)
    target_repr = target_repr.to(device)
    
    print("Calculating loss")
    loss = F.mse_loss(pred_repr * combined_mask, target_repr * combined_mask)
    print(f"Calculated loss: {loss.item()}")
    
    loss.backward()
    optimizer.step()
    
    if step % 500 == 0:
        print_jet_details(pred_repr[0].cpu(), "Predicted")
        visualize_predictions_vs_ground_truth(subjets[0].cpu(), pred_repr[0].cpu(), title=f"Ground Truth vs Predictions (Step {step})")
        print(f"Context representation shape: {context_repr.shape}")
        print(f"Target representation shape: {target_repr.shape}")
        
    return loss.item()

if __name__ == "__main__":
    print("Starting main program")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    try:
        print("Loading dataset")
        train_dataset = JetDataset("../data/val/val_20_30.h5", subset_size=1000, config=config)
    except Exception as e:
        print(f"Error loading dataset: {e}")

    print("Creating DataLoader")
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=custom_collate_fn)

    print("Initializing model")
    model = JJEPA(input_dim=240, embed_dim=512, depth=12, num_heads=8, mlp_ratio=4.0, dropout=0.1).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.04)

    num_epochs = 10
    train_losses = []
    
    print(f"Starting training for {num_epochs} epochs")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{num_epochs}", leave=True, position=0)
    
        for step, (particle_features, subjet_features, particle_indices, subjet_mask, particle_mask) in enumerate(train_loader):
            subjet_particles = particle_features[:, :, particle_indices]
            features = features.to(device)
            subjets = subjets.to(device)
            subjet_masks = subjet_masks.to(device)
            particle_masks = particle_masks.to(device)
            
            loss = train_step(model, subjets, subjet_masks, particle_masks, optimizer, device, step)
            total_loss += loss
            
            progress_bar.set_postfix(loss=loss)
            progress_bar.update(1)
            
            if step % 100 == 0:
                print(f"\nEpoch {epoch+1}, Step {step}, Loss: {loss:.4f}")
        
        progress_bar.close()

        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

    print("Training completed")
    print("Visualizing training loss")
    visualize_training_loss(train_losses)

    print("Saving model")
    torch.save(model.state_dict(), 'ijepa_model.pth')

    print("Model saved.")
