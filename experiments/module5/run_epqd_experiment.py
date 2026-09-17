def train_epqd_single_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int = 15,
    max_steps: Optional[int] = None,
) -> Tuple[List[float], List[float], float]:
    """Train single EPQD model and return train losses, val losses, and total train time."""

    model.train()

    train_loss_history = []
    val_loss_history = []

    t_start = time.time()
    total_steps = 0

    for epoch in range(epochs):

        print(
            f"        Epoch {epoch + 1}/{epochs} starting...",
            flush=True,
        )

        epoch_loss = 0.0
        n_batches = 0

        model.train()

        for x_t_batch, x_0_batch, t_batch in train_loader:

            x_t_batch = x_t_batch.to(device)
            x_0_batch = x_0_batch.to(device)
            t_batch = t_batch.to(device)

            optimizer.zero_grad()

            x_pred = model(x_t_batch, t_batch)

            loss = F.mse_loss(
                x_pred,
                x_0_batch,
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            total_steps += 1

            if max_steps is not None and total_steps >= max_steps:
                break

        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_loss_history.append(avg_train_loss)

        # ---------------------------------------------------------------
        # Validation pass
        # ---------------------------------------------------------------

        model.eval()

        val_loss = 0.0
        val_batches = 0

        with torch.no_grad():

            for vx_t, vx_0, vt in val_loader:

                vx_t = vx_t.to(device)
                vx_0 = vx_0.to(device)
                vt = vt.to(device)

                v_pred = model(vx_t, vt)

                val_loss += F.mse_loss(
                    v_pred,
                    vx_0,
                ).item()

                val_batches += 1

        avg_val_loss = val_loss / max(val_batches, 1)
        val_loss_history.append(avg_val_loss)

        print(
            f"        Epoch {epoch + 1}/{epochs} complete | "
            f"Train MSE: {avg_train_loss:.6f} | "
            f"Val MSE: {avg_val_loss:.6f} | "
            f"Steps: {total_steps}",
            flush=True,
        )

        if max_steps is not None and total_steps >= max_steps:
            break

    total_time = time.time() - t_start

    print(
        f"        Training complete | "
        f"Time: {total_time:.2f}s | "
        f"Steps: {total_steps}",
        flush=True,
    )

    return (
        train_loss_history,
        val_loss_history,
        total_time,
    )