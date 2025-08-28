import os
import sys
sys.path.append("..")
import torch
import logging
from typing import Union, Callable, Tuple, List

from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, \
    average_precision_score, accuracy_score, precision_score,f1_score,recall_score
from sklearn.neighbors import KNeighborsClassifier

from .loss import *
from .model import * 
from .utils import one_hot_encoding


def Trainer(
        model: torch.nn.Module,
        model_optimizer: torch.optim.Optimizer,
        classifier: torch.nn.Module,
        classifier_optimizer: torch.optim.Optimizer,
        train_dl: torch.utils.data.DataLoader,
        valid_dl: torch.utils.data.DataLoader,
        test_dl: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
        logger: logging.Logger,
        config,
        experiment_log_dir: str,
        training_mode: str
    ):
    """
    The main training function

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    model_optimizer : torch.optim.Optimizer
        The optimizer for the model.
    classifier: torch.nn.Module
        The classifier for downstream task.
    classifier_optimizer: torch.optim.Optimizer
        The optimizer for the classifier.
    train_dl, valid_dl, test_dl : DataLoader
        The training, validation, and test data loaders.
        The train_dl is used for both pre-training and fine-tuning.
        The valid_dl and test_dl are only used in fine-tuning and testing.
    device : torch.device or str
        The device to run the training on (CPU or GPU).
    logger : logging.Logger
        The logger for logging training progress.
    config : object
        The configuration parameters
        - num_epoch: int, number of epochs
        - batch_size: int, batch size
        - lr: float, learning rate
        - beta1: float, beta1 for Adam optimizer
        - beta2: float, beta2 for Adam optimizer
        - weight_decay: float, weight decay for Adam optimizer
        - Context_Cont: object, context contrastive learning parameters
            - temperature: float, temperature for NTXentLoss
            - use_cosine_similarity: bool, whether to use cosine similarity
            - lambda_tf: float, weight for temporal-frequency contrastive loss
    experiment_log_dir : str
        The directory to save experiment logs and models.
    training_mode : str
        The training mode, either 'pre_train' or 'fine_tune_test'.
    """
    # Start training
    logger.debug("Training started ....")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    if training_mode == 'pre_train':
        print('Pretraining on source dataset')
        for epoch in range(1, config.num_epoch + 1):
            # Train and validate
            nt_xent_criterion = NTXentLoss_poly(
                device, config.batch_size, config.Context_Cont.temperature,
                config.Context_Cont.use_cosine_similarity)
            train_loss = model_pretrain(
                model, model_optimizer, nt_xent_criterion, train_dl, device)
            logger.debug(f'\nPre-training Epoch {epoch}: Training Loss -- {train_loss:.4f}')

        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = {'model_state_dict': model.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))
        print('Pretrained model is stored at location:{}'.format(experiment_log_dir+'saved_models'+'ckp_last.pt'))

    elif training_mode == 'fine_tune_test':
        model = model.to(device)
        classifier = classifier.to(device)
        print('Loading the pre-trained model')
        model.load_state_dict(torch.load(os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))['model_state_dict'])

        print('Fine-tune on Fine-tuning set')
        performance_list = []
        total_f1 = []
        KNN_f1 = []
        global emb_finetune, label_finetune, emb_test, label_test

        for epoch in range(1, config.num_epoch + 1):
            logger.debug(f'\nEpoch : {epoch}')

            valid_loss, emb_finetune, label_finetune, F1 = model_finetune(
                model, model_optimizer, valid_dl, config,
                device, training_mode,
                classifier=classifier, classifier_optimizer=classifier_optimizer)

            scheduler.step(valid_loss)

            # save best fine-tuning model""
            global arch
            arch = 'sleepedf2eplipsy'
            if len(total_f1) == 0 or F1 > max(total_f1):
                print('update fine-tuned model')
                os.makedirs('experiments_logs/finetunemodel/', exist_ok=True)
                torch.save(model.state_dict(), 'experiments_logs/finetunemodel/' + arch + '_model.pt')
                torch.save(classifier.state_dict(), 'experiments_logs/finetunemodel/' + arch + '_classifier.pt')
            total_f1.append(F1)

            # evaluate on the test set
            """Testing set"""
            logger.debug('Test on Target datasts test set')
            model.load_state_dict(torch.load('experiments_logs/finetunemodel/' + arch + '_model.pt'))
            classifier.load_state_dict(torch.load('experiments_logs/finetunemodel/' + arch + '_classifier.pt'))
            _, emb_test, label_test, performance = model_test(model, test_dl, device, classifier)
            performance_list.append(performance)

            """Use KNN as another classifier; it's an alternation of the MLP classifier in function model_test. 
            Experiments show KNN and MLP may work differently in different settings, so here we provide both. """
            # train classifier: KNN
            neigh = KNeighborsClassifier(n_neighbors=5)
            neigh.fit(emb_finetune, label_finetune)
            knn_acc_train = neigh.score(emb_finetune, label_finetune)
            # print('KNN finetune acc:', knn_acc_train)
            representation_test = emb_test.detach().cpu().numpy()

            knn_result = neigh.predict(representation_test)
            knn_result_score = neigh.predict_proba(representation_test)
            one_hot_label_test = one_hot_encoding(label_test)
            # print(classification_report(label_test, knn_result, digits=4))
            # print(confusion_matrix(label_test, knn_result))
            knn_acc = accuracy_score(label_test, knn_result)
            precision = precision_score(label_test, knn_result, average='macro', )
            recall = recall_score(label_test, knn_result, average='macro', )
            F1 = f1_score(label_test, knn_result, average='macro')
            auc = roc_auc_score(one_hot_label_test, knn_result_score, average="macro", multi_class="ovr")
            prc = average_precision_score(one_hot_label_test, knn_result_score, average="macro")
            print('KNN Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f'%
                    (knn_acc, precision, recall, F1, auc, prc))
            KNN_f1.append(F1)

        logger.debug("\n################## Best testing performance! #########################")
        performance_array = np.array(performance_list)
        best_performance = performance_array[np.argmax(performance_array[:,0], axis=0)]
        print('Best Testing Performance: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f '
                '| AUPRC=%.4f' % (best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                                best_performance[4], best_performance[5]))
        print('Best KNN F1', max(KNN_f1))

    else:
        raise ValueError("Invalid training mode. Choose either 'pre_train' or 'fine_tune_test'.")

    logger.debug("\n################## Training is Done! #########################")


def model_pretrain(
        model: torch.nn.Module,
        model_optimizer: torch.optim.Optimizer,
        criterion: Callable,
        train_loader: torch.utils.data.DataLoader,
        device: Union[torch.device, str],
    ) -> torch.Tensor:
    """
    Pre-training the model with contrastive learning:
    temporal contrastive loss, frequency contrastive loss, temporal-frequency consistency loss.

    Parameters
    ----------
    model : torch.nn.Module
        The model to be trained.
    model_optimizer : torch.optim.Optimizer
        The optimizer for the model.
    criterion : Callable
        The loss function for contrastive learning.
    train_loader : DataLoader
        The training data loader.
    device : torch.device or str
        The device to run the training on (CPU or GPU).

    Returns
    -------
    torch.Tensor
        The average training loss for the epoch.
    """
    total_loss = []
    model.train()
    global loss, loss_t, loss_f, l_TF, loss_c

    # optimizer
    model_optimizer.zero_grad()

    for data, labels, aug1, data_f, aug1_f in train_loader:
        data, labels = data.float().to(device), labels.long().to(device)
        aug1 = aug1.float().to(device)
        data_f, aug1_f = data_f.float().to(device), aug1_f.float().to(device)

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)

        """Compute Pre-train loss"""
        loss_t = criterion(h_t, h_t_aug)
        loss_f = criterion(h_f, h_f_aug)
        l_TF = criterion(z_t, z_f)  # temporal-frequency consistency loss

        l_1, l_2, l_3 = criterion(z_t, z_f_aug), criterion(z_t_aug, z_f), criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3)   # consistency triplet loss

        lam = 0.2
        loss = lam * (loss_t + loss_f) + l_TF

        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()

    print('Pretraining loss breakdown: temporal contrastive -- {}, '
          'frequency contrastive -- {}, consistency loss -- {}'.format(loss, loss_t, loss_f, l_TF))

    ave_loss = torch.tensor(total_loss).mean()

    return ave_loss


def model_finetune(
        model: torch.nn.Module,
        model_optimizer: torch.optim.Optimizer,
        val_dl: torch.utils.data.DataLoader,
        config,
        device: Union[str, torch.device],
        training_mode: str,
        classifier: torch.nn.Module,
        classifier_optimizer: torch.optim.Optimizer
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, Union[float, np.ndarray]]:
    """
    Finetune a classifier on the downstream dataset
    with metrics: accuracy, precision, recall, F1, AUROC, AUPRC.

    Returns
    -------
    ave_loss: torch.Tensor
        The average validation loss for the epoch. (scalar)
    feas: np.ndarray
        The learned embeddings for the validation set. (num_samples, embedding_dim)
    trgs: np.ndarray
        The true labels for the validation set. (num_samples,)
    F1: np.ndarray or float
        F1 score of the positive class in binary classification or
        weighted average of the F1 scores of each class for the multiclass task.
    """
    global labels, pred_numpy, fea_concat_flat
    model.train()
    classifier.train()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    feas = np.array([])

    for data, labels, aug1, data_f, aug1_f in val_dl:
        # print('Fine-tuning: {} of target samples'.format(labels.shape[0]))
        data, labels = data.float().to(device), labels.long().to(device)
        data_f = data_f.float().to(device)
        aug1 = aug1.float().to(device)
        aug1_f = aug1_f.float().to(device)

        """if random initialization:"""
        model_optimizer.zero_grad()  # The gradients are zero, but the parameters are still randomly initialized.
        classifier_optimizer.zero_grad()  # the classifier is newly added and randomly initialized

        """Produce embeddings"""
        h_t, z_t, h_f, z_f = model(data, data_f)
        h_t_aug, z_t_aug, h_f_aug, z_f_aug = model(aug1, aug1_f)
        nt_xent_criterion = NTXentLoss_poly(device, config.target_batch_size, config.Context_Cont.temperature,
                                            config.Context_Cont.use_cosine_similarity)
        loss_t = nt_xent_criterion(h_t, h_t_aug)
        loss_f = nt_xent_criterion(h_f, h_f_aug)
        l_TF = nt_xent_criterion(z_t, z_f)

        l_1, l_2, l_3 = nt_xent_criterion(z_t, z_f_aug), nt_xent_criterion(z_t_aug, z_f), \
                        nt_xent_criterion(z_t_aug, z_f_aug)
        loss_c = (1 + l_TF - l_1) + (1 + l_TF - l_2) + (1 + l_TF - l_3) #


        """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test."""
        fea_concat = torch.cat((z_t, z_f), dim=1)
        predictions = classifier(fea_concat)
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        loss_p = criterion(predictions, labels)

        lam = 0.1
        loss = loss_p + l_TF + lam*(loss_t + loss_f)

        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
        onehot_label = F.one_hot(labels)
        pred_numpy = predictions.detach().cpu().numpy()

        try:
            auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr" )
        except:
            auc_bs = 0.
        prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)

        total_acc.append(acc_bs)
        total_auc.append(auc_bs)
        total_prc.append(prc_bs)
        total_loss.append(loss.item())
        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        if training_mode != "pre_train":
            pred = predictions.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            feas = np.append(feas, fea_concat_flat.data.cpu().numpy())

    feas = feas.reshape([len(trgs), -1])  # produce the learned embeddings

    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)
    precision = precision_score(labels_numpy, pred_numpy, average='macro', )
    recall = recall_score(labels_numpy, pred_numpy, average='macro', )
    F1 = f1_score(labels_numpy, pred_numpy, average='macro', )
    if not isinstance(F1, np.ndarray):
        F1 = float(F1)

    ave_loss = torch.tensor(total_loss).mean()
    ave_acc = torch.tensor(total_acc).mean()
    ave_auc = torch.tensor(total_auc).mean()
    ave_prc = torch.tensor(total_prc).mean()

    print(' Finetune: loss = %.4f| Acc=%.4f | Precision = %.4f | Recall = %.4f | F1 = %.4f| AUROC=%.4f | AUPRC = %.4f'
          % (ave_loss, ave_acc*100, precision * 100, recall * 100, F1 * 100, ave_auc * 100, ave_prc *100))

    return ave_loss, feas, trgs, F1


def model_test(
        model: torch.nn.Module,
        test_dl: torch.utils.data.DataLoader,
        device: Union[str, torch.device],
        classifier: torch.nn.Module
    ) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, List[str]]:
    """
    Test a feature extractor (model) and a classifier on the test set
    using metrics: accuracy, precision, recall, F1, AUROC, AUPRC.

    Returns
    -------
    total_loss: torch.Tensor
        The average test loss for the epoch. (scalar)
    emb_test_all: torch.Tensor
        The learned embeddings for the test set. (num_samples, embedding_dim)
    trgs: np.ndarray
        The true labels for the test set. (num_samples,)
    performance: List[float]
        list of performance metrics: [acc, precision, recall, F1, AUROC, AUPRC]
        in percentage form (0-100)
    """
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []

    criterion = nn.CrossEntropyLoss() # the loss for downstream classifier
    outs = np.array([])
    trgs = np.array([])
    emb_test_all = []

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels, _,data_f, _ in test_dl:
            data, labels = data.float().to(device), labels.long().to(device)
            data_f = data_f.float().to(device)

            """Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test"""
            _, z_t, _, z_f = model(data, data_f)
            fea_concat = torch.cat((z_t, z_f), dim=1)
            predictions_test = classifier(fea_concat)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            emb_test_all.append(fea_concat_flat)

            loss = criterion(predictions_test, labels)
            acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(labels)
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()
            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy,
                                   average="macro", multi_class="ovr")
            except:
                auc_bs = 0.
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro")
            pred_numpy = np.argmax(pred_numpy, axis=1)

            total_acc.append(acc_bs)
            total_auc.append(auc_bs)
            total_prc.append(prc_bs)

            total_loss.append(loss.item())
            pred = predictions_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())
            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))

    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    # print('Test classification report', classification_report(labels_numpy_all, pred_numpy_all))
    # print(confusion_matrix(labels_numpy_all, pred_numpy_all))
    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()

    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]
    print('MLP Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f | AUROC= %.4f | AUPRC=%.4f'
          % (acc*100, precision * 100, recall * 100, F1 * 100, total_auc*100, total_prc*100))
    emb_test_all = torch.concat(tuple(emb_test_all))
    return total_loss, emb_test_all, trgs, performance
