# Métodos Computacionais aplicados à Geofísica (MCOM)

Disciplina obrigatória ministrada ao Programa de Pós-Graduação em Geofísica do Observatório Nacional (ON/MCTI).

**Professor**: [André Luis A. Reis](https://sipos.on.br/geofisica/frontend/docentes/index.php?idd=57)

**E-mails:** reisandreluis@on.br / andre.reis@gmail.com

>**Aviso:** O material disponibilizado neste repositório está em constante desenvolvimento e, portanto, o Observatório Nacional e a coordenação de pós-graduação não possuem qualquer responsabilidade sobre este conteúdo. As aulas não serão gravadas. Todo o material didático e computacional está localizado na pasta 'Content'.

## Ementa

Primeiros passos em Python. Aplicação: Filtro de média móvel simples. Operações matriciais básicas. Matrizes especiais. Solução numérica de sistemas lineares: Introdução a sistemas lineares - sistemas lineares especiais - Eliminação Gaussiana - Decomposição LU - Decomposição LDLT - Decomposição de Cholesky - Mínimos quadrados. Aplicação: Rede gravimétrica simples. Solução numérica de sistemas não-lineares: Método de Newton, Método da descida mais íngreme, Método de Gauss-Newton, Método de Levenberg-Marquardt. Aplicação: Estimativa das coordenadas de um epicentro. Interpolação: Método de Lagrange, Método de Neville, Ajuste polinomial. Aplicação: Gridagem de uma anomalia de gravidade. Solução numérica de equações diferenciais: Diferenças finitas. Aplicação: Simulação de um decaimento exponencial simples. Integração numérica: Fórmulas de Newton-Cotes, Quadratura Gaussiana. Aplicação: Simulação de uma perfilagem sísmica vertical. Transformadas: Transformada de Fourier, Transformada de Hilbert. Aplicação: Amplitude do Sinal Analítico 2D de um perfil de anomalia de campo total.

Versão oficial do conteúdo da disciplina: [Métodos Computacionais aplicado à Geofísica](https://www.gov.br/observatorio/pt-br/assuntos/programas-academicos/pos-graduacao-em-geofisica/documentos/ementas/metodos_computacionais_aplicados_a_geofisica_rev.pdf)

## Tópicos do curso

* Operações básicas de Vetores e Matrizes
* Soluções de sistemas lineares
* Soluções de sistemas não lineares
* Transformada de Fourier
* Interpolação numérica
* Integração numéricas
* Aplicações na Geofísica

## Conteúdo didático computacional

>**Aviso:** Os códigos aqui apresentados são parte de uma disciplina e sua usabilidade é, consideravelmente, limitada a nível de pesquisa e desenvolvimento. A instituição não tem qualquer responsabilidade sobre a aplicação dos códigos aqui apresentados, tanto a nível acadêmico quanto profissional.

**Introdução**
  - [ ] Notação para vetores e matrizes [`0. notation.ipynb`]

**1. Operações com vetores**
  - [ ] Produto escalar-vetor [`1a. escalar-vetor.ipynb`]
  - [ ] Tempo de execução de funções [`1b. tempo-escalar-vetor.ipynb`]
  - [ ] Produto escalar entre vetores [`1c. dot-product.ipynb`]
  - [ ] Produto externo entre vetores [`1d. outer-product.ipynb`]
  - [ ] Produto elemento a elemento [`1e. hadamard-product.ipynb`]

**2. Operações com matrizes**
  - [ ] Produto matriz-vetor [`2a. matrix-vector.ipynb`]
  - [ ] Produto matriz-matriz [`2b. matrix-matrix.ipynb`]   

**3. FLOPS**
  - [ ] Flops [`3. flops.ipynb`]

**4. Normas de vetores e matrizes**
  - [ ] Normas de vetores [`4a. vector-norm.ipynb`]
  - [ ] Normas matriciais [`4b. matrix-norm.ipynb`]

**5. Estruturas matriciais**
  - [ ] Matriz diagonal [`5a. diagonal-matrices.ipynb`]
  - [ ] Matriz triangular [`5b. triangular-matrices.ipynb`]
  - [ ] Matriz em bloco [`5c. block-matrices.ipynb`]
  - [ ] Matriz de Permutação [`5d. permutation-matrices.ipynb`]

**6. Transformada de Fourier**

**7. Solução numérica de sistemas lineares**

**8. Autovalores e autovetores**

**9. Decomposição em valores singulares**

**10. Soluções numéricas de sistemas não-lineares**

**11. Interpolação e ajuste de curva**

**12. Derivação numérica**

**13. Integração numérica**

## Referências bibliográficas

* Aster, R. C., Borchers, B., and Thurber, C. H. 2005. *Parameter Estimation and Inverse Problems*. Academic Press Inc.

* Bard, Y. 1974. *Nonlinear parameter estimation*. Academic Press Inc.

* Golub, G. H. and Van Loan, C. F. 2013. *Matrix computations*. Johns Hopkins University Press.

* Kelley, C. T. 1999. *Iterative methods for optimization: Raleigh*. SIAM.

* Kiusalaas, J. 2013. *Numerical methods in engineering with Python 3*. Cambridge University Press.

* Menke, W. 1989. *Geophysical data analysis: Discrete inverse theory*. Academic Press Inc.

* Parker, R. L. 1977. *Understanding inverse theory*. Ann. Rev.Earth. Planet. Sci., v. 5, p. 35-64.

## Material

Todo o material da disciplina (dados e códigos computacionais) estão disponíveis em um repositório no Github:

https://github.com/andrelreis/metodos-computacionais

As versões ao final de cada ano são marcadas com um *tag* e podem ser vistas em:

https://github.com/andrelreis/metodos-computacionais/releases


## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br /><span xmlns:dct="http://purl.org/dc/terms/" href="http://purl.org/dc/dcmitype/Text" property="dct:title" rel="dct:type">"Material didático da disciplina Métodos Computacionais aplicados à Geofísica"</span>
by <a xmlns:cc="http://creativecommons.org/ns#" href="https://github.com/andrelreis/metodos-potenciais" property="cc:attributionName" rel="cc:attributionURL">André L A Reis</a> is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
