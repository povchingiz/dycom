"""
labkit.theory — мост между теорией на бумаге и экспериментом.

Здесь живёт то, что отличает твою лабораторию от книжных ноутбуков:
вывод loss из MLE и доверительные интервалы через гессиан (информацию Фишера).
Книга подводит вплотную (глава 5, задача 6.2) — этот модуль доводит до конца.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------------------
# MLE -> loss. Каждая loss ВЫВОДИТСЯ из предположения о распределении шума.
# Это не выбор из меню, а следствие модели данных.
# ---------------------------------------------------------------------------

def gaussian_nll(y_true, y_pred, sigma=1.0):
    """Гауссов шум  ->  негативный log-likelihood  ->  (с точностью до констант) MSE.
    Предположение: y = f(x) + eps, eps ~ N(0, sigma^2).
    """
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    resid = y_true - y_pred
    n = resid.size
    return (n / 2) * np.log(2 * np.pi * sigma**2) + np.sum(resid**2) / (2 * sigma**2)


def laplace_nll(y_true, y_pred, b=1.0):
    """Лапласовский шум  ->  NLL  ->  MAE. (Книга этого не выводит — твоя добавка.)"""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = y_true.size
    return n * np.log(2 * b) + np.sum(np.abs(y_true - y_pred)) / b


def bernoulli_nll(y_true, p_pred, eps=1e-12):
    """Бернуллиевы метки  ->  NLL  ->  binary cross-entropy."""
    y_true, p_pred = np.asarray(y_true), np.clip(np.asarray(p_pred), eps, 1 - eps)
    return -np.sum(y_true * np.log(p_pred) + (1 - y_true) * np.log(1 - p_pred))


# ---------------------------------------------------------------------------
# Доверительные интервалы через кривизну log-likelihood.
# Гессиан NLL в минимуме = наблюдаемая информация Фишера.
# Обратная Фишера = ковариация оценки. Корень из диагонали = SE.
# ---------------------------------------------------------------------------

def hessian_fd(nll_fn, theta, eps=1e-5):
    """Численный гессиан NLL по параметрам theta (конечные разности)."""
    theta = np.asarray(theta, dtype=float)
    d = theta.size
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            tpp, tpm, tmp, tmm = (theta.copy() for _ in range(4))
            tpp[i] += eps; tpp[j] += eps
            tpm[i] += eps; tpm[j] -= eps
            tmp[i] -= eps; tmp[j] += eps
            tmm[i] -= eps; tmm[j] -= eps
            H[i, j] = (nll_fn(tpp) - nll_fn(tpm) - nll_fn(tmp) + nll_fn(tmm)) / (4 * eps**2)
    return H


def confidence_intervals(nll_fn, theta_hat, level=0.95):
    """CI на параметры через информацию Фишера.
    Возвращает (lower, upper, se). theta_hat — оценка максимального правдоподобия.
    """
    from scipy.stats import norm
    H = hessian_fd(nll_fn, theta_hat)          # наблюдаемая информация Фишера
    cov = np.linalg.inv(H)                     # ковариация оценки
    se = np.sqrt(np.diag(cov))
    z = norm.ppf(0.5 + level / 2)
    theta_hat = np.asarray(theta_hat, dtype=float)
    return theta_hat - z * se, theta_hat + z * se, se
