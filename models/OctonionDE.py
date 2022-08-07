#OctonionDE
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from .Model import Model
from numpy.random import RandomState

class OctonionDE(Model):
    def __init__(self, config):
        super(OctonionDE, self).__init__(config)
        #8 embeddings for head/tail
        self.emb_1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_3 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_4 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_5 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_6 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_7 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.emb_8 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        #8 embeddings for relation
        self.rel_1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_3 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_4 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_5 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_6 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_7 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_8 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        #8 relation transfer embeddings 
        self.rel_transfer1 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_transfer2 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_transfer3 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_transfer4 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_transfer5 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_transfer6 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_transfer7 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        self.rel_transfer8 = nn.Embedding(self.config.relTotal, self.config.hidden_size)
        #8 head/tail transfer embeddings
        self.ent_transfer1 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ent_transfer2 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ent_transfer3 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ent_transfer4 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ent_transfer5 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ent_transfer6 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ent_transfer7 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.ent_transfer8 = nn.Embedding(self.config.entTotal, self.config.hidden_size)
        self.criterion = nn.Softplus()
        self.init_weights()

    def init_weights(self):
        #init weight for head/tail embeddings
        nn.init.xavier_uniform_(self.emb_1.weight.data)
        nn.init.xavier_uniform_(self.emb_2.weight.data)
        nn.init.xavier_uniform_(self.emb_3.weight.data)
        nn.init.xavier_uniform_(self.emb_4.weight.data)
        nn.init.xavier_uniform_(self.emb_5.weight.data)
        nn.init.xavier_uniform_(self.emb_6.weight.data)
        nn.init.xavier_uniform_(self.emb_7.weight.data)
        nn.init.xavier_uniform_(self.emb_8.weight.data)
        #init weight for relation embeddings
        nn.init.xavier_uniform_(self.rel_1.weight.data)
        nn.init.xavier_uniform_(self.rel_2.weight.data)
        nn.init.xavier_uniform_(self.rel_3.weight.data)
        nn.init.xavier_uniform_(self.rel_4.weight.data)
        nn.init.xavier_uniform_(self.rel_5.weight.data)
        nn.init.xavier_uniform_(self.rel_6.weight.data)
        nn.init.xavier_uniform_(self.rel_7.weight.data)
        nn.init.xavier_uniform_(self.rel_8.weight.data)
        #init weight for relation transfer embeddings
        nn.init.xavier_uniform_(self.rel_transfer1.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer2.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer3.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer4.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer5.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer6.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer7.weight.data)
        nn.init.xavier_uniform_(self.rel_transfer8.weight.data)
        #init weight for head/tail transfer embeddings
        nn.init.xavier_uniform_(self.ent_transfer1.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer2.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer3.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer4.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer5.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer6.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer7.weight.data)
        nn.init.xavier_uniform_(self.ent_transfer8.weight.data)

    def _qmult(self, s_a, x_a, y_a, z_a, s_b, x_b, y_b, z_b):
        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a
        return A, B, C, D

    def _qstar(self, a, b, c, d):
        return a, -b, -c, -d

    def _omult(self, a_1, a_2, a_3, a_4, b_1, b_2, b_3, b_4, c_1, c_2, c_3, c_4, d_1, d_2, d_3, d_4):
        d_1_star, d_2_star, d_3_star, d_4_star = self._qstar(d_1, d_2, d_3, d_4)
        c_1_star, c_2_star, c_3_star, c_4_star = self._qstar(c_1, c_2, c_3, c_4)
        o_1, o_2, o_3, o_4 = self._qmult(a_1, a_2, a_3, a_4, c_1, c_2, c_3, c_4 )
        o_1s, o_2s, o_3s, o_4s = self._qmult(d_1_star, d_2_star, d_3_star, d_4_star,  b_1, b_2, b_3, b_4)
        o_5, o_6, o_7, o_8 = self._qmult(d_1, d_2, d_3, d_4, a_1, a_2, a_3, a_4 )
        o_5s, o_6s, o_7s, o_8s = self._qmult(b_1, b_2, b_3, b_4, c_1_star, c_2_star, c_3_star, c_4_star)
        return  o_1 - o_1s, o_2 - o_2s, o_3 - o_3s, o_4 - o_4s, o_5 + o_5s, o_6 + o_6s, o_7 + o_7s, o_8 + o_8s

    def _onorm(self, r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8):
        denominator = torch.sqrt(r_1**2 + r_2**2 + r_3**2 + r_4**2 + r_5**2 + r_6**2 + r_7**2 + r_8**2)
        r_1 = r_1 / denominator
        r_2 = r_2 / denominator
        r_3 = r_3 / denominator
        r_4 = r_4 / denominator
        r_5 = r_5 / denominator
        r_6 = r_6 / denominator
        r_7 = r_7 / denominator
        r_8 = r_8 / denominator
        return  r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8

    def _calc(self, e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                    e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
                    r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8, 
                    h_transfer1, h_transfer2, h_transfer3, h_transfer4, h_transfer5, h_transfer6, h_transfer7, h_transfer8,
                    t_transfer1, t_transfer2, t_transfer3, t_transfer4, t_transfer5, t_transfer6, t_transfer7, t_transfer8,
                    r_transfer1, r_transfer2, r_transfer3, r_transfer4, r_transfer5, r_transfer6, r_transfer7, r_transfer8):
        r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8 = self._onorm(r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)
        h_transfer1, h_transfer2, h_transfer3, h_transfer4, h_transfer5, h_transfer6, h_transfer7, h_transfer8 = self._onorm(h_transfer1, h_transfer2, h_transfer3, h_transfer4,
                                                                                                                             h_transfer5, h_transfer6, h_transfer7, h_transfer8)
        t_transfer1, t_transfer2, t_transfer3, t_transfer4, t_transfer5, t_transfer6, t_transfer7, t_transfer8 = self._onorm(t_transfer1, t_transfer2, t_transfer3, t_transfer4,
                                                                                                                             t_transfer5, t_transfer6, t_transfer7, t_transfer8)
        r_transfer1, r_transfer2, r_transfer3, r_transfer4, r_transfer5, r_transfer6, r_transfer7, r_transfer8 = self._onorm(r_transfer1, r_transfer2, r_transfer3, r_transfer4,
                                                                                                                             r_transfer5, r_transfer6, r_transfer7, r_transfer8)
        h_h_transfer1, h_h_transfer2, h_h_transfer3, h_h_transfer4, h_h_transfer5, h_h_transfer6, h_h_transfer7, h_h_transfer8 = self._omult(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                                                                                            h_transfer1, h_transfer2, h_transfer3, h_transfer4, h_transfer5, h_transfer6, h_transfer7, h_transfer8)
        h_h_r_transfer1, h_h_r_transfer2, h_h_r_transfer3, h_h_r_transfer4, h_h_r_transfer5, h_h_r_transfer6, h_h_r_transfer7, h_h_r_transfer8 = self._omult(h_h_transfer1, h_h_transfer2, h_h_transfer3, 
                                                                                                                                                             h_h_transfer4, h_h_transfer5, h_h_transfer6, 
                                                                                                                                                             h_h_transfer7, h_h_transfer8, r_transfer1, 
                                                                                                                                                             r_transfer2, r_transfer3, r_transfer4,
                                                                                                                                                    r_transfer5, r_transfer6, r_transfer7, r_transfer8)
        t_t_transfer1, t_t_transfer2, t_t_transfer3, t_t_transfer4, t_t_transfer5, t_t_transfer6, t_t_transfer7, t_t_transfer8 = self._omult(e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
                                                                                                t_transfer1, t_transfer2, t_transfer3, t_transfer4, t_transfer5, t_transfer6, t_transfer7, t_transfer8)
        t_t_r_transfer1, t_t_r_transfer2, t_t_r_transfer3, t_t_r_transfer4, t_t_r_transfer5, t_t_r_transfer6, t_t_r_transfer7, t_t_r_transfer8 = self._omult(t_t_transfer1, t_t_transfer2, t_t_transfer3,
                                                                                                                                                             t_t_transfer4, t_t_transfer5, t_t_transfer6,
                                                                                                                                                             t_t_transfer7, t_t_transfer8, r_transfer1,
                                                                                                                                                             r_transfer2, r_transfer3, r_transfer4,
                                                                                                                                                      r_transfer5, r_transfer6, r_transfer7, r_transfer8)
        o_1, o_2, o_3, o_4, o_5, o_6, o_7, o_8 = self._omult(h_h_r_transfer1, h_h_r_transfer2, h_h_r_transfer3, h_h_r_transfer4, 
                                                             h_h_r_transfer5, h_h_r_transfer6, h_h_r_transfer7, h_h_r_transfer8,
                                                             r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8)
        score_r = (o_1 * t_t_r_transfer1 + o_2 * t_t_r_transfer2 + o_3 * t_t_r_transfer3 + o_4 * t_t_r_transfer4 + 
                   o_5 * t_t_r_transfer5 + o_6 * t_t_r_transfer6 + o_7 * t_t_r_transfer7 + o_8 * t_t_r_transfer8)
        return -torch.sum(score_r, -1)

    def loss(self, score, regul):
        return (torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul)

    def forward(self):
        #8 embeddings of head
        e_1_h = self.emb_1(self.batch_h)
        e_2_h = self.emb_2(self.batch_h)
        e_3_h = self.emb_3(self.batch_h)
        e_4_h = self.emb_4(self.batch_h)
        e_5_h = self.emb_5(self.batch_h)
        e_6_h = self.emb_6(self.batch_h)
        e_7_h = self.emb_7(self.batch_h)
        e_8_h = self.emb_8(self.batch_h)
        #8 embeddings of tail
        e_1_t = self.emb_1(self.batch_t)
        e_2_t = self.emb_2(self.batch_t)
        e_3_t = self.emb_3(self.batch_t)
        e_4_t = self.emb_4(self.batch_t)
        e_5_t = self.emb_5(self.batch_t)
        e_6_t = self.emb_6(self.batch_t)
        e_7_t = self.emb_7(self.batch_t)
        e_8_t = self.emb_8(self.batch_t)
        #8 embeddings of relation
        r_1 = self.rel_1(self.batch_r)
        r_2 = self.rel_2(self.batch_r)
        r_3 = self.rel_3(self.batch_r)
        r_4 = self.rel_4(self.batch_r)
        r_5 = self.rel_5(self.batch_r)
        r_6 = self.rel_6(self.batch_r)
        r_7 = self.rel_7(self.batch_r)
        r_8 = self.rel_8(self.batch_r)
        #8 head transfer embeddings
        h_transfer1 = self.ent_transfer1(self.batch_h)
        h_transfer2 = self.ent_transfer2(self.batch_h)
        h_transfer3 = self.ent_transfer3(self.batch_h)
        h_transfer4 = self.ent_transfer4(self.batch_h)
        h_transfer5 = self.ent_transfer5(self.batch_h)
        h_transfer6 = self.ent_transfer6(self.batch_h)
        h_transfer7 = self.ent_transfer7(self.batch_h)
        h_transfer8 = self.ent_transfer8(self.batch_h)
        #8 head transfer embeddings
        t_transfer1 = self.ent_transfer1(self.batch_t)
        t_transfer2 = self.ent_transfer2(self.batch_t)
        t_transfer3 = self.ent_transfer3(self.batch_t)
        t_transfer4 = self.ent_transfer4(self.batch_t)
        t_transfer5 = self.ent_transfer5(self.batch_t)
        t_transfer6 = self.ent_transfer6(self.batch_t)
        t_transfer7 = self.ent_transfer7(self.batch_t)
        t_transfer8 = self.ent_transfer8(self.batch_t)
        #8 relation transfer embeddings
        r_transfer1 = self.rel_transfer1(self.batch_r)
        r_transfer2 = self.rel_transfer2(self.batch_r)
        r_transfer3 = self.rel_transfer3(self.batch_r)
        r_transfer4 = self.rel_transfer4(self.batch_r)
        r_transfer5 = self.rel_transfer5(self.batch_r)
        r_transfer6 = self.rel_transfer6(self.batch_r)
        r_transfer7 = self.rel_transfer7(self.batch_r)
        r_transfer8 = self.rel_transfer8(self.batch_r)
        #calc score
        score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                           e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
                           r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8,
                           h_transfer1, h_transfer2, h_transfer3, h_transfer4, h_transfer5, h_transfer6, h_transfer7, h_transfer8,
                           t_transfer1, t_transfer2, t_transfer3, t_transfer4, t_transfer5, t_transfer6, t_transfer7, t_transfer8,
                           r_transfer1, r_transfer2, r_transfer3, r_transfer4, r_transfer5, r_transfer6, r_transfer7, r_transfer8)
        regul = (torch.mean(torch.abs(e_1_h) ** 2) 
                 + torch.mean(torch.abs(e_2_h) ** 2)
                 + torch.mean(torch.abs(e_3_h) ** 2)
                 + torch.mean(torch.abs(e_4_h) ** 2)
                 + torch.mean(torch.abs(e_5_h) ** 2)
                 + torch.mean(torch.abs(e_6_h) ** 2)
                 + torch.mean(torch.abs(e_7_h) ** 2)
                 + torch.mean(torch.abs(e_8_h) ** 2)
                 + torch.mean(torch.abs(e_1_t) ** 2)
                 + torch.mean(torch.abs(e_2_t) ** 2)
                 + torch.mean(torch.abs(e_3_t) ** 2)
                 + torch.mean(torch.abs(e_4_t) ** 2)
                 + torch.mean(torch.abs(e_5_t) ** 2)
                 + torch.mean(torch.abs(e_6_t) ** 2)
                 + torch.mean(torch.abs(e_7_t) ** 2)
                 + torch.mean(torch.abs(e_8_t) ** 2)
                 + torch.mean(torch.abs(r_1) ** 2)
                 + torch.mean(torch.abs(r_2) ** 2)
                 + torch.mean(torch.abs(r_3) ** 2)
                 + torch.mean(torch.abs(r_4) ** 2)
                 + torch.mean(torch.abs(r_5) ** 2)
                 + torch.mean(torch.abs(r_6) ** 2)
                 + torch.mean(torch.abs(r_7) ** 2)
                 + torch.mean(torch.abs(r_8) ** 2)
                 + torch.mean(torch.abs(h_transfer1) ** 2)
                 + torch.mean(torch.abs(h_transfer2) ** 2)
                 + torch.mean(torch.abs(h_transfer3) ** 2)
                 + torch.mean(torch.abs(h_transfer4) ** 2)
                 + torch.mean(torch.abs(h_transfer5) ** 2)
                 + torch.mean(torch.abs(h_transfer6) ** 2)
                 + torch.mean(torch.abs(h_transfer7) ** 2)
                 + torch.mean(torch.abs(h_transfer8) ** 2)
                 + torch.mean(torch.abs(t_transfer1) ** 2)
                 + torch.mean(torch.abs(t_transfer2) ** 2)
                 + torch.mean(torch.abs(t_transfer3) ** 2)
                 + torch.mean(torch.abs(t_transfer4) ** 2)
                 + torch.mean(torch.abs(t_transfer5) ** 2)
                 + torch.mean(torch.abs(t_transfer6) ** 2)
                 + torch.mean(torch.abs(t_transfer7) ** 2)
                 + torch.mean(torch.abs(t_transfer8) ** 2)
                 + torch.mean(torch.abs(r_transfer1) ** 2)
                 + torch.mean(torch.abs(r_transfer2) ** 2)
                 + torch.mean(torch.abs(r_transfer3) ** 2)
                 + torch.mean(torch.abs(r_transfer4) ** 2)
                 + torch.mean(torch.abs(r_transfer5) ** 2)
                 + torch.mean(torch.abs(r_transfer6) ** 2)
                 + torch.mean(torch.abs(r_transfer7) ** 2)
                 + torch.mean(torch.abs(r_transfer8) ** 2))
        return self.loss(score, regul) 

    def predict(self):
        #8 embeddings of head
        e_1_h = self.emb_1(self.batch_h)
        e_2_h = self.emb_2(self.batch_h)
        e_3_h = self.emb_3(self.batch_h)
        e_4_h = self.emb_4(self.batch_h)
        e_5_h = self.emb_5(self.batch_h)
        e_6_h = self.emb_6(self.batch_h)
        e_7_h = self.emb_7(self.batch_h)
        e_8_h = self.emb_8(self.batch_h)
        #8 embeddings of tail
        e_1_t = self.emb_1(self.batch_t)
        e_2_t = self.emb_2(self.batch_t)
        e_3_t = self.emb_3(self.batch_t)
        e_4_t = self.emb_4(self.batch_t)
        e_5_t = self.emb_5(self.batch_t)
        e_6_t = self.emb_6(self.batch_t)
        e_7_t = self.emb_7(self.batch_t)
        e_8_t = self.emb_8(self.batch_t)
        #8 embeddings of relation
        r_1 = self.rel_1(self.batch_r)
        r_2 = self.rel_2(self.batch_r)
        r_3 = self.rel_3(self.batch_r)
        r_4 = self.rel_4(self.batch_r)
        r_5 = self.rel_5(self.batch_r)
        r_6 = self.rel_6(self.batch_r)
        r_7 = self.rel_7(self.batch_r)
        r_8 = self.rel_8(self.batch_r)
        #8 head transfer embeddings
        h_transfer1 = self.ent_transfer1(self.batch_h)
        h_transfer2 = self.ent_transfer2(self.batch_h)
        h_transfer3 = self.ent_transfer3(self.batch_h)
        h_transfer4 = self.ent_transfer4(self.batch_h)
        h_transfer5 = self.ent_transfer5(self.batch_h)
        h_transfer6 = self.ent_transfer6(self.batch_h)
        h_transfer7 = self.ent_transfer7(self.batch_h)
        h_transfer8 = self.ent_transfer8(self.batch_h)
        #8 head transfer embeddings
        t_transfer1 = self.ent_transfer1(self.batch_t)
        t_transfer2 = self.ent_transfer2(self.batch_t)
        t_transfer3 = self.ent_transfer3(self.batch_t)
        t_transfer4 = self.ent_transfer4(self.batch_t)
        t_transfer5 = self.ent_transfer5(self.batch_t)
        t_transfer6 = self.ent_transfer6(self.batch_t)
        t_transfer7 = self.ent_transfer7(self.batch_t)
        t_transfer8 = self.ent_transfer8(self.batch_t)
        #8 relation transfer embeddings
        r_transfer1 = self.rel_transfer1(self.batch_r)
        r_transfer2 = self.rel_transfer2(self.batch_r)
        r_transfer3 = self.rel_transfer3(self.batch_r)
        r_transfer4 = self.rel_transfer4(self.batch_r)
        r_transfer5 = self.rel_transfer5(self.batch_r)
        r_transfer6 = self.rel_transfer6(self.batch_r)
        r_transfer7 = self.rel_transfer7(self.batch_r)
        r_transfer8 = self.rel_transfer8(self.batch_r)
        #calc score
        score = self._calc(e_1_h, e_2_h, e_3_h, e_4_h, e_5_h, e_6_h, e_7_h, e_8_h,
                           e_1_t, e_2_t, e_3_t, e_4_t, e_5_t, e_6_t, e_7_t, e_8_t,
                           r_1, r_2, r_3, r_4, r_5, r_6, r_7, r_8,
                           h_transfer1, h_transfer2, h_transfer3, h_transfer4, h_transfer5, h_transfer6, h_transfer7, h_transfer8,
                           t_transfer1, t_transfer2, t_transfer3, t_transfer4, t_transfer5, t_transfer6, t_transfer7, t_transfer8,
                           r_transfer1, r_transfer2, r_transfer3, r_transfer4, r_transfer5, r_transfer6, r_transfer7, r_transfer8)
        return score.cpu().data.numpy()