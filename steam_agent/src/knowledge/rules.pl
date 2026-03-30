% =========================================
% CAPA 1: DERIVADOS BÁSICOS
% =========================================

% Un usuario jugó un juego si tiene interacción
played(U, G) :-
    performedBy(I, U),
    involvesGame(I, G).

% Alto engagement (más de 10 horas)
high_engagement(I) :-
    PlaySession(I),
    duration(I, D),
    D > 10.

% =========================================
% CAPA 2: PREFERENCIAS
% =========================================

% Preferencia por uso
likes_tag(U, T) :-
    performedBy(I, U),
    involvesGame(I, G),
    hasTag(G, T),
    high_engagement(I).

% Preferencia por rating positivo
likes_tag(U, T) :-
    Rating(I),
    ratingValue(I, "Recomendado"),
    performedBy(I, U),
    involvesGame(I, G),
    hasTag(G, T).

% =========================================
% CAPA 3: SIMILARIDAD
% =========================================

% Juegos similares si comparten al menos 2 tags
similar_game(G1, G2) :-
    G1 \= G2,
    hasTag(G1, T1),
    hasTag(G2, T1),
    hasTag(G1, T2),
    hasTag(G2, T2),
    T1 \= T2.

% =========================================
% CAPA 4: SUBGÉNEROS
% =========================================

subgenre(G, t) :-
    hasTag(G, t).

% =========================================
% CAPA 5: CANDIDATOS
% =========================================

% Por preferencias
candidate(U, G) :-
    likes_tag(U, T),
    hasTag(G, T).

% Por similaridad
candidate(U, G2) :-
    played(U, G1),
    similar_game(G1, G2).

% Filtrar juegos ya jugados
valid_candidate(U, G) :-
    candidate(U, G),
    \+ played(U, G).

% =========================================
% CAPA 6: EVENTOS (DESCUENTOS)
% =========================================

discounted_game(G) :-
    hasDiscount(G, D),
    D > 0.3.

candidate(U, G) :-
    discounted_game(G).

% =========================================
% CAPA 7: RECOMENDACIÓN FINAL
% =========================================

recommend(U, G) :-
    valid_candidate(U, G).

% =========================================
% CAPA 8: EXPLICACIONES
% =========================================

% Explicación por tags
explanation(U, G, reason_tag(T)) :-
    recommend(U, G),
    likes_tag(U, T),
    hasTag(G, T).

% Explicación por similaridad
explanation(U, G, reason_similar(G2)) :-
    recommend(U, G),
    played(U, G2),
    similar_game(G2, G).

% Explicación por subgénero
explanation(U, G, reason_subgenre(S)) :-
    recommend(U, G),
    subgenre(G, S).

% Explicación por descuento
explanation(U, G, reason_discount) :-
    recommend(U, G),
    discounted_game(G).

% =========================================
% REGLAS AUXILIARES
% =========================================

% Evitar que un juego sea similar a sí mismo
not_similar(G) :-
    \+ similar_game(G, G).
