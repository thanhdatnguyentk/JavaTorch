import re
with open('core/src/main/resources/public/index.html', 'r', encoding='utf-8') as f:
    content = f.read()

stats_html = '''
                <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8" v-if="sysStats">
                    <div class="bg-slate-900/40 p-4 rounded-xl border border-white/5">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1"><i data-lucide="cpu" class="w-3 h-3"></i> VRAM Used</div>
                        <div class="text-lg font-mono text-accent">{{ sysStats.processUsedMB.toFixed(1) }} MB</div>
                    </div>
                    <div class="bg-slate-900/40 p-4 rounded-xl border border-white/5">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1"><i data-lucide="database" class="w-3 h-3"></i> Pool Used</div>
                        <div class="text-lg font-mono text-emerald-400">{{ sysStats.poolUsedMB.toFixed(1) }} MB</div>
                    </div>
                    <div class="bg-slate-900/40 p-4 rounded-xl border border-white/5">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1"><i data-lucide="hard-drive" class="w-3 h-3"></i> VRAM Total</div>
                        <div class="text-lg font-mono text-purple-400">{{ sysStats.totalMB.toFixed(1) }} MB</div>
                    </div>
                    <div class="bg-slate-900/40 p-4 rounded-xl border border-white/5">
                        <div class="text-xs text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1"><i data-lucide="clock" class="w-3 h-3"></i> Last Update</div>
                        <div class="text-lg font-mono text-slate-300">{{ lastUpdate }}</div>
                    </div>
                </div>

                <!-- Chart Container -->'''

content = content.replace('<!-- Chart Container -->', stats_html)

setup_vars = '''
            const sysStats = ref(null);
            const lastUpdate = ref("-");
            const inferenceTask = ref('classify_image');'''

content = content.replace("const inferenceTask = ref('classify_image');", setup_vars)

ws_onmessage = '''ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        console.log("WS received:", data);

                        if (data.vram) {
                            sysStats.value = data.vram;
                        }
                        if (data.timestamp) {
                            const d = new Date(data.timestamp);
                            lastUpdate.value = d.toLocaleTimeString() + "." + d.getMilliseconds().toString().padStart(3, '0');
                        }'''

content = content.replace('''ws.onmessage = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        console.log("WS received:", data);''', ws_onmessage)


ret_statement = '''return {
                currentTab, epochs, metricKeys, getMetricValue, wsConnected,
                sysStats, lastUpdate,'''

content = content.replace('return {\n                currentTab, epochs, metricKeys, getMetricValue, wsConnected,', ret_statement)

with open('core/src/main/resources/public/index.html', 'w', encoding='utf-8') as f:
    f.write(content)
