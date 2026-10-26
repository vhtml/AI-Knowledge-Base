<!-- frontend/src/App.vue -->
<script setup>
import { ref } from 'vue'
const input=ref('')
const messages=ref([])
const loading=ref(false)

async function send(){
 const text=input.value.trim()
 if(!text || loading.value) return
 messages.value.push({role:'user',content:text})
 input.value=''; loading.value=true
 try{
  const r=await fetch('http://localhost:8000/chat',{
   method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({message:text})
  })
  if(!r.ok) throw new Error('请求失败：'+r.status)
  const data=await r.json()
  messages.value.push({role:'assistant',content:data.answer})
 }catch(e){
  messages.value.push({role:'error',content:e.message})
 }finally{ loading.value=false }
}
</script>

<template>
 <main>
  <h1>我的 Agent</h1>
  <div v-for="(m,i) in messages" :key="i" :class="m.role">
   <b>{{m.role==='user'?'我':'Agent'}}：</b>{{m.content}}
  </div>
  <p v-if="loading">Agent 正在处理…</p>
  <form @submit.prevent="send">
   <input v-model="input" placeholder="输入任务" />
   <button :disabled="loading">发送</button>
  </form>
 </main>
</template>