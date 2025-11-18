import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from PIL import Image
import os
from dotenv import load_dotenv

load_dotenv()

# Configuração da página
st.set_page_config(page_title="FoodCopy AI", page_icon="🍔", layout="centered")

st.title("🍔 FoodCopy AI")
st.markdown("#### Gere descrições apetitosas para cardápios do iFood em segundos!")

# Modelo atual
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
        groq_api_key=os.getenv("GROQ_API_KEY")
    )

llm = get_llm()

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """Você é um copywriter especialista em gastronomia do iFood.
Escreva descrições extremamente apetitosas, sensoriais e vendedoras.
Use palavras que ativam fome: suculento, crocante, derretido, artesanal, fresquinho, irresistível...
Máximo 2 linhas, até 280 caracteres.
Termine sempre com uma chamada sutil para o pedido."""),
    ("user", """Prato: {nome_prato}
Ingredientes/observações: {ingredientes}

Gere 3 variações numeradas (1, 2, 3) bem diferentes entre si.""")
])

chain = prompt | llm

# Interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Informações do prato")
    nome_prato = st.text_input("Nome do prato", placeholder="Ex: X-Tudo, Açaí 500ml")
    ingredientes = st.text_area("Ingredientes ou detalhes (quanto mais detalhe, melhor!)", height=120)

with col2:
    st.subheader("Foto do prato (opcional - só exibição)")
    uploaded_file = st.file_uploader("Mostre o prato!", type=["png", "jpg", "jpeg", "webp"])

# Exibe a foto somente de visualização
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    if image.mode in ("RGBA", "P", "LA", "CMYK"):
        image = image.convert("RGB")
    st.image(image, caption="Prato carregado! Descreva na caixa de ingredientes o que você vê para descrições perfeitas.", use_container_width=True)

# Botão gerar
if st.button("✨ Gerar 3 descrições vendedoras", type="primary", use_container_width=True):
    if not nome_prato.strip():
        st.error("Coloque pelo menos o nome do prato!")
    else:
        with st.spinner("Llama 3.3 70B criando textos irresistíveis..."):
            resposta = chain.invoke({
                "nome_prato": nome_prato,
                "ingredientes": ingredientes or "Sem detalhes"
            }).content

            st.success("Pronto! Aqui estão suas 3 descrições:")
            st.markdown(resposta)
            st.code(resposta, language=None)

st.markdown("---")