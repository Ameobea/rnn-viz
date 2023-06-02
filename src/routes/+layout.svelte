<script lang="ts">
  import { maybeInitSentry } from '../sentry';

  import 'src/index.css';

  maybeInitSentry();
</script>

<slot />
