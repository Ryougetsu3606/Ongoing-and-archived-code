# Description

**Failed** trial for Speech Enhancement Knowledge Distillation Training, with a Bilevel Optimization Framework.

$$\min_\phi L_{GT}(\theta,\phi)=\ell_{MSE}(\hat{y},y),$$ $$\text{s.t. }\theta\in\arg \min_{\theta'} L_{KD}(\theta',\phi) = \sum_i \ell_{MSE}(z_{T_i},z_{S_i})$$


$\theta$ for encoder, $\phi$ for decoder. $y$ for clean speech, $\hat{y}$ for enhanced speech. $z_T$ for latent representation of teacher model while $z_S$ for student model.



