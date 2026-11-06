"""verl 0.9.1 reward-loop adapter (not the older workers reward API)."""
from verl.experimental.reward_loop.reward_manager.naive import NaiveRewardManager

class StrictNumericRewardManager(NaiveRewardManager):
    async def run_single(self, data):
        data = data[-1:]
        data.non_tensor_batch = dict(data.non_tensor_batch)
        info = data.non_tensor_batch['extra_info'].copy()
        info[0] = dict(info[0])
        data.non_tensor_batch['extra_info'] = info
        item = data[0]
        responses = item.batch["responses"]
        count = int(item.batch["attention_mask"][-responses.shape[-1]:].sum().item())
        eos = self.tokenizer.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        ended = count > 0 and int(responses[count-1].item()) in eos
        # Extra info lives on this one response; never infer completion from text.
        item.non_tensor_batch["extra_info"]["dagrpo_truncated"] = not ended
        return await super().run_single(data)
